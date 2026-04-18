#pragma once
// ============================================================================
// kawpow_proggen.hpp — KAWPOW random-program JIT source generator
//
// Precisely mirrors the RNG sequence from cpp-kawpow's progpow.cpp round().
// The generated code is injected into kawpow.hip at /* PROGPOW_PROGRAM */
// before RTC compilation.
//
// Usage:
//   std::string src = kernel_template;
//   src = kawpow_proggen::inject_coin_padding(src, KAWPOW_PADDING_RVN);
//   src = kawpow_proggen::inject_program(src, block_number);
//   // RTC compile src
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <string>
#include <array>
#include <algorithm>

#include <ethash/kawpow_coins.h>

namespace kawpow_proggen {

// Constants (must match kawpow.hip AND cpp-kawpow reference)
static constexpr uint32_t NUM_REGS           = 32;
static constexpr uint32_t NUM_LANES          = 16;
static constexpr uint32_t CNT_CACHE          = 11;
static constexpr uint32_t CNT_MATH           = 18;
static constexpr uint32_t NUM_WORDS_PER_LANE = 4;   // sizeof(hash2048) / (4 * 16)
static constexpr uint32_t FNV_PRIME          = 0x01000193u;
static constexpr uint32_t FNV_OFFSET_BASIS   = 0x811c9dc5u;

// ---------------------------------------------------------------------------
// Host-side FNV-1a
// ---------------------------------------------------------------------------
inline uint32_t fnv1a(uint32_t h, uint32_t d) {
    return (h ^ d) * FNV_PRIME;
}

// ---------------------------------------------------------------------------
// Host-side KISS99 — identical to cpp-kawpow kiss99.hpp
// ---------------------------------------------------------------------------
struct Kiss99 {
    uint32_t z, w, jsr, jcong;

    uint32_t operator()() {
        z = 36969u * (z & 0xffffu) + (z >> 16);
        w = 18000u * (w & 0xffffu) + (w >> 16);
        jcong = 69069u * jcong + 1234567u;
        jsr ^= (jsr << 17);
        jsr ^= (jsr >> 13);
        jsr ^= (jsr << 5);
        return (((z << 16) + w) ^ jcong) + jsr;
    }
};

// ---------------------------------------------------------------------------
// Host-side mix_rng_state — mirrors cpp-kawpow progpow.cpp::mix_rng_state
//
// CRITICAL: dst and src are shuffled in the SAME loop, interleaving RNG calls.
// The reference does: for each i, swap dst[i], then swap src[i].
// Sequences wrap with modulo, never re-shuffle.
// ---------------------------------------------------------------------------
struct MixRngState {
    Kiss99 rng;
    std::array<uint32_t, NUM_REGS> dst_seq;
    std::array<uint32_t, NUM_REGS> src_seq;
    uint32_t dst_counter = 0;
    uint32_t src_counter = 0;

    explicit MixRngState(uint64_t period) {
        uint32_t seed_lo = static_cast<uint32_t>(period);
        uint32_t seed_hi = static_cast<uint32_t>(period >> 32);

        uint32_t z    = fnv1a(FNV_OFFSET_BASIS, seed_lo);
        uint32_t w    = fnv1a(z, seed_hi);
        uint32_t jsr  = fnv1a(w, seed_lo);
        uint32_t jcong = fnv1a(jsr, seed_hi);
        rng = {z, w, jsr, jcong};

        for (uint32_t i = 0; i < NUM_REGS; ++i) {
            dst_seq[i] = i;
            src_seq[i] = i;
        }

        // Fisher-Yates — interleaved dst/src, matching reference exactly
        for (uint32_t i = NUM_REGS; i > 1; --i) {
            std::swap(dst_seq[i - 1], dst_seq[rng() % i]);
            std::swap(src_seq[i - 1], src_seq[rng() % i]);
        }
    }

    uint32_t next_dst() { return dst_seq[(dst_counter++) % NUM_REGS]; }
    uint32_t next_src() { return src_seq[(src_counter++) % NUM_REGS]; }
};

// ---------------------------------------------------------------------------
// Code emission helpers — inline the operations for better GPU perf
// ---------------------------------------------------------------------------

// Emit merge: a = f(a, b, sel)
// Must match cpp-kawpow random_merge() exactly:
//   case 0: a = (a * 33) + b
//   case 1: a = (a ^ b) * 33
//   case 2: a = rotl32(a, x) ^ b    where x = (sel>>16)%31 + 1
//   case 3: a = rotr32(a, x) ^ b
inline std::string emit_merge(const std::string& a, const std::string& b, uint32_t sel) {
    uint32_t x = ((sel >> 16) % 31) + 1;
    switch (sel % 4) {
    case 0: return a + " = (" + a + " * 33u) + " + b + ";";
    case 1: return a + " = (" + a + " ^ " + b + ") * 33u;";
    case 2: return a + " = rotl32(" + a + ", " + std::to_string(x) + "u) ^ " + b + ";";
    case 3: return a + " = rotr32(" + a + ", " + std::to_string(x) + "u) ^ " + b + ";";
    }
    return {};
}

// Emit math: result = f(a, b, sel)
// Must match cpp-kawpow random_math() exactly
inline std::string emit_math(const std::string& a, const std::string& b, uint32_t sel) {
    switch (sel % 11) {
    case  0: return a + " + " + b;
    case  1: return a + " * " + b;
    case  2: return "__umulhi(" + a + ", " + b + ")";
    case  3: return "min(" + a + ", " + b + ")";
    case  4: return "rotl32(" + a + ", " + b + ")";
    case  5: return "rotr32(" + a + ", " + b + ")";
    case  6: return a + " & " + b;
    case  7: return a + " | " + b;
    case  8: return a + " ^ " + b;
    case  9: return "__clz(" + a + ") + __clz(" + b + ")";
    case 10: return "__popc(" + a + ") + __popc(" + b + ")";
    }
    return {};
}

inline std::string reg(uint32_t r) { return "mix[" + std::to_string(r) + "]"; }

// ---------------------------------------------------------------------------
// Fused math+merge for compound instruction patterns.
// Returns empty string if no fusion applies (caller falls back to _m temp).
//
// IMPORTANT: never strength-reduce * 33u here — the compiler does it better.
// Only fuse when the combined expression exposes a multi-operand ISA pattern
// that the _m temp would obscure.
//
// v_add3_u32 patterns (three-way add, AMD RDNA + NVIDIA):
//   math(add:0)       + merge(mul33+add:0): (dst*33) + a + b
//   math(clz+clz:9)   + merge(mul33+add:0): (dst*33) + __clz(a) + __clz(b)
//   math(popc+popc:10) + merge(mul33+add:0): (dst*33) + __popc(a) + __popc(b)
//
// v_xor3_b32 patterns (three-way xor, AMD RDNA + NVIDIA):
//   math(xor:8) + merge(rotl^:2):    rotl32(dst, x) ^ a ^ b
//   math(xor:8) + merge(rotr^:3):    rotr32(dst, x) ^ a ^ b
//   math(xor:8) + merge(xor_mul33:1): (dst ^ a ^ b) * 33
//
// lop3.b32 patterns (NVIDIA only — 3-input boolean in 1 instruction):
//   math(and:6) + merge(rotl^:2/rotr^:3): rot(dst,x) ^ (a & b)  → lop3 0x78
//   math(or:7)  + merge(rotl^:2/rotr^:3): rot(dst,x) ^ (a | b)  → lop3 0x1E
//   math(and:6) + merge(xor_mul33:1):     (dst ^ (a & b)) * 33   → lop3 0x78
//   math(or:7)  + merge(xor_mul33:1):     (dst ^ (a | b)) * 33   → lop3 0x1E
// ---------------------------------------------------------------------------

// Helper: emit NVIDIA lop3.b32 inline asm, guarded by #ifndef __HIP_PLATFORM_AMD__
// lop3.b32 computes an arbitrary 3-input boolean function in 1 instruction.
// truth_table: 8-bit LUT index (e.g. 0x78 = P^(Q&R), 0x1E = P^(Q|R))
inline std::string emit_lop3(const std::string& dst,
                              const std::string& a,
                              const std::string& b,
                              const std::string& c,
                              uint32_t truth_table) {
    char tt[8];
    std::snprintf(tt, sizeof(tt), "0x%02x", truth_table);
    // NVIDIA path: single lop3 instruction
    // AMD path: expand to individual ops (compiler picks best encoding)
    return "{ uint32_t _t;\n"
           "#ifndef __HIP_PLATFORM_AMD__\n"
           "        asm(\"lop3.b32 %0, %1, %2, %3, " + std::string(tt) + ";\" "
           ": \"=r\"(_t) : \"r\"((uint32_t)" + a + "), \"r\"((uint32_t)" + b + "), \"r\"((uint32_t)" + c + "));\n"
           "#else\n"
           "        _t = " + a + (truth_table == 0x78 ? " ^ (" + b + " & " + c + ")" :
                                  truth_table == 0x1E ? " ^ (" + b + " | " + c + ")" :
                                  " ^ " + b + " ^ " + c) + ";\n"
           "#endif\n"
           "        " + dst + " = _t; }";
}

inline std::string try_fuse_math_merge(const std::string& dst,
                                       const std::string& src1,
                                       const std::string& src2,
                                       uint32_t math_sel,
                                       uint32_t merge_sel) {
    uint32_t math_op  = math_sel % 11;
    uint32_t merge_op = merge_sel % 4;

    // ---- v_add3 patterns: merge(0) with add-like math ----
    if (merge_op == 0) {
        if (math_op == 0) {
            return dst + " = (" + dst + " * 33u) + " + src1 + " + " + src2 + ";";
        }
        if (math_op == 9) {
            return dst + " = (" + dst + " * 33u) + __clz(" + src1 + ") + __clz(" + src2 + ");";
        }
        if (math_op == 10) {
            return dst + " = (" + dst + " * 33u) + __popc(" + src1 + ") + __popc(" + src2 + ");";
        }
    }

    // ---- v_xor3 patterns: math(xor:8) with xor-like merge ----
    if (math_op == 8) {
        if (merge_op == 1) {
            return dst + " = (" + dst + " ^ " + src1 + " ^ " + src2 + ") * 33u;";
        }
        if (merge_op == 2) {
            uint32_t x = ((merge_sel >> 16) % 31) + 1;
            return dst + " = rotl32(" + dst + ", " + std::to_string(x) + "u) ^ " + src1 + " ^ " + src2 + ";";
        }
        if (merge_op == 3) {
            uint32_t x = ((merge_sel >> 16) % 31) + 1;
            return dst + " = rotr32(" + dst + ", " + std::to_string(x) + "u) ^ " + src1 + " ^ " + src2 + ";";
        }
    }

    // ---- lop3 patterns: math(and:6 / or:7) with xor-based merge ----
    // NVIDIA: lop3.b32 computes P^(Q&R) or P^(Q|R) in 1 instruction
    // AMD: falls back to 2 separate ops (and/or + xor)
    if (math_op == 6 || math_op == 7) {
        uint32_t tt = (math_op == 6) ? 0x78u : 0x1Eu; // and→0x78, or→0x1E

        // and/or + rotl_xor: lop3(rotl32(dst, x), src1, src2)
        if (merge_op == 2) {
            uint32_t x = ((merge_sel >> 16) % 31) + 1;
            return emit_lop3(dst, "rotl32(" + dst + ", " + std::to_string(x) + "u)",
                             src1, src2, tt);
        }
        // and/or + rotr_xor: lop3(rotr32(dst, x), src1, src2)
        if (merge_op == 3) {
            uint32_t x = ((merge_sel >> 16) % 31) + 1;
            return emit_lop3(dst, "rotr32(" + dst + ", " + std::to_string(x) + "u)",
                             src1, src2, tt);
        }
        // and/or + xor_mul33: lop3(dst, src1, src2) * 33
        if (merge_op == 1) {
            return "{ uint32_t _t;\n"
                   "#ifndef __HIP_PLATFORM_AMD__\n"
                   "        asm(\"lop3.b32 %0, %1, %2, %3, " +
                   std::string(tt == 0x78 ? "0x78" : "0x1e") + ";\" "
                   ": \"=r\"(_t) : \"r\"((uint32_t)" + dst + "), \"r\"((uint32_t)" + src1 + "), \"r\"((uint32_t)" + src2 + "));\n"
                   "#else\n"
                   "        _t = " + dst + (tt == 0x78 ? " ^ (" + src1 + " & " + src2 + ")" :
                                            " ^ (" + src1 + " | " + src2 + ")") + ";\n"
                   "#endif\n"
                   "        " + dst + " = _t * 33u; }";
        }
    }

    return {}; // no fusion — use _m temp
}

// ---------------------------------------------------------------------------
// generate_program — produce the inner-loop body for a given block number
//
// This replays the EXACT same RNG sequence as cpp-kawpow's round() function.
// The round() function is called with mix_rng_state passed BY VALUE, so
// every outer-loop iteration runs the same program.
// ---------------------------------------------------------------------------
inline std::string generate_program(int block_number) {
    uint64_t period = static_cast<uint64_t>(block_number) / 3;
    MixRngState state(period);

    char buf[256];
    snprintf(buf, sizeof(buf),
        "        // ProgPoW program for period %llu (block %d)\n",
        (unsigned long long)period, block_number);

    // ---- Phase 1a: collect all ops (RNG order preserved exactly) ----
    struct CacheOp { uint32_t src, dst, sel; };
    struct MathOp  { uint32_t src1, src2, sel1, dst, sel2; };
    CacheOp cache_ops[CNT_CACHE];
    MathOp  math_ops[CNT_MATH];

    constexpr int max_ops = (CNT_CACHE > CNT_MATH) ? CNT_CACHE : CNT_MATH;

    for (int i = 0; i < max_ops; ++i) {
        if (i < (int)CNT_CACHE) {
            cache_ops[i] = { state.next_src(), state.next_dst(), state.rng() };
        }
        if (i < (int)CNT_MATH) {
            uint32_t sr = state.rng() % (NUM_REGS * (NUM_REGS - 1));
            uint32_t s1 = sr % NUM_REGS, s2 = sr / NUM_REGS;
            if (s2 >= s1) ++s2;
            uint32_t sel1 = state.rng();
            uint32_t mdst = state.next_dst();
            uint32_t sel2 = state.rng();
            math_ops[i] = { s1, s2, sel1, mdst, sel2 };
        }
    }

    // ---- Phase 1b: emit with ILP cache read pairing ----
    // When safe, issue two LDS reads back-to-back so the compiler can
    // overlap them (LDS has 2-cycle latency, dual-issue hides it).
    // Safe when: cache[i+1].src is NOT written by cache[i].merge or math[i].

    // Helper lambda to emit a math op block
    auto emit_math_block = [&](std::string& out, const MathOp& m) {
        std::string fused = try_fuse_math_merge(reg(m.dst), reg(m.src1), reg(m.src2), m.sel1, m.sel2);
        if (!fused.empty()) {
            out += "        " + fused + "\n";
        } else {
            out += "        _m = " + emit_math(reg(m.src1), reg(m.src2), m.sel1) + ";\n";
            out += "        " + emit_merge(reg(m.dst), "_m", m.sel2) + "\n";
        }
    };

    std::string body;
    body.reserve(6144);
constexpr int MAX_K = 4;

auto can_pair_k = [&](int i, int k) -> bool {
    if (i + k > (int)CNT_CACHE) return false;
    for (int j = 1; j < k; ++j) {
        uint32_t s = cache_ops[i + j].src;
        for (int p = 0; p < j; ++p) {
            if (s == cache_ops[i + p].dst) return false;
            if (i + p < (int)CNT_MATH && s == math_ops[i + p].dst) return false;
        }
    }
    return true;
};

// ensure we've declared enough temps at body scope
body += "        uint32_t _c0, _c1, _c2, _c3, _m, _m1;\n";

int i = 0;
while (i < max_ops) {
    int k = 1;
    if (i < (int)CNT_CACHE) {
        int kmax = std::min(MAX_K, (int)CNT_CACHE - i);
        for (int t = kmax; t >= 2; --t) {
            if (can_pair_k(i, t)) { k = t; break; }
        }
    }

    if (k >= 2) {
        // Phase 1: issue all k LDS reads up front
        for (int j = 0; j < k; ++j) {
            body += "        _c" + std::to_string(j) + " = l1_cache["
                  + reg(cache_ops[i + j].src) + " & 0xFFFu];\n";
        }
        // Phase 2: interleave merges and maths in original order
        for (int j = 0; j < k; ++j) {
            body += "        " + emit_merge(reg(cache_ops[i + j].dst),
                                            "_c" + std::to_string(j),
                                            cache_ops[i + j].sel) + "\n";
            if (i + j < (int)CNT_MATH) emit_math_block(body, math_ops[i + j]);
        }
        i += k;
    } else if (i >= (int)CNT_CACHE && i + 1 < (int)CNT_MATH) {
        // Tail: pure math ops — try to pair adjacent ops for VOPD packing
        // Safe when math[i+1] doesn't read math[i].dst
        const auto& m0 = math_ops[i];
        const auto& m1 = math_ops[i + 1];
        // Safe when: m1 doesn't read m0.dst, m0 doesn't read m1.dst,
        // and they don't write the same dst (merge reads+writes dst)
        bool can_pair = (m1.src1 != m0.dst && m1.src2 != m0.dst
                      && m0.src1 != m1.dst && m0.src2 != m1.dst
                      && m0.dst != m1.dst);
        if (can_pair) {
            // Check if either can fuse — if so, emit individually (fuse is already optimal)
            std::string fused0 = try_fuse_math_merge(reg(m0.dst), reg(m0.src1), reg(m0.src2), m0.sel1, m0.sel2);
            std::string fused1 = try_fuse_math_merge(reg(m1.dst), reg(m1.src1), reg(m1.src2), m1.sel1, m1.sel2);
            if (fused0.empty() && fused1.empty()) {
                // Neither fuses — pair them: compute both, then merge both
                body += "        _m = " + emit_math(reg(m0.src1), reg(m0.src2), m0.sel1) + ";\n";
                body += "        _m1 = " + emit_math(reg(m1.src1), reg(m1.src2), m1.sel1) + ";\n";
                body += "        " + emit_merge(reg(m0.dst), "_m", m0.sel2) + "\n";
                body += "        " + emit_merge(reg(m1.dst), "_m1", m1.sel2) + "\n";
                i += 2;
            } else {
                // At least one fuses — emit individually
                emit_math_block(body, m0);
                ++i;
            }
        } else {
            // Dependency — emit individually
            emit_math_block(body, m0);
            ++i;
        }
    } else {
        // 1-way fallback
        if (i < (int)CNT_CACHE) {
            body += "        _c0 = l1_cache[" + reg(cache_ops[i].src) + " & 0xFFFu];\n";
            body += "        " + emit_merge(reg(cache_ops[i].dst), "_c0", cache_ops[i].sel) + "\n";
        }
        if (i < (int)CNT_MATH) emit_math_block(body, math_ops[i]);
        ++i;
    }
}

    // ---- Phase 2: generate DAG params (RNG advances AFTER cache+math) ----
    uint32_t dag_dsts[NUM_WORDS_PER_LANE];
    uint32_t dag_sels[NUM_WORDS_PER_LANE];
    for (uint32_t i = 0; i < NUM_WORDS_PER_LANE; ++i) {
        dag_dsts[i] = (i == 0) ? 0 : state.next_dst();
        dag_sels[i] = state.rng();
    }

    // ---- Phase 3: assemble in execution order ----
    // DAG load FIRST (compiler will issue the global_load_dwordx4 here,
    // then schedule all the VALU/LDS from cache+math before inserting
    // s_waitcnt vmcnt(0) at the merge point)

    std::string c;
    c.reserve(8192);
    c += buf; // header comment
    c += "\n";

    // DAG load — issued early, data arrives during cache+math
    // Nontemporal (SLC) to avoid L2 pollution from random 4.6GB DAG reads
    c += "        uint4 _dg = ((const uint4*)d_dag)"
        "[dag_addr * 16u + ((lane_id ^ loop) & 15u)];\n\n";

    // Cache + math body
    c += body;

    // DAG merge — compiler inserts s_waitcnt vmcnt(0) right here
    c += "\n        // DAG merge\n";
    static const char* fields[] = {"_dg.x", "_dg.y", "_dg.z", "_dg.w"};
    for (uint32_t i = 0; i < NUM_WORDS_PER_LANE; ++i) {
        c += "        " + emit_merge(reg(dag_dsts[i]), fields[i], dag_sels[i]) + "\n";
    }

    return c;
}

// ---------------------------------------------------------------------------
// generate_program_glc — GLC variant with inline asm DAG load
//
// Same RNG sequence and body as generate_program(), but uses non-volatile
// asm for global_load_dwordx4 with GLC flag.  The body MUST be emitted
// directly inside the main loop — NOT in a separate function.
// ---------------------------------------------------------------------------
inline std::string generate_program_glc(int block_number) {
    uint64_t period = static_cast<uint64_t>(block_number) / 3;
    MixRngState state(period);

    char buf[256];
    snprintf(buf, sizeof(buf),
        "        // ProgPoW GLC program for period %llu (block %d)\n",
        (unsigned long long)period, block_number);

    // ---- Phase 1a: collect all ops (RNG order preserved exactly) ----
    struct CacheOp { uint32_t src, dst, sel; };
    struct MathOp  { uint32_t src1, src2, sel1, dst, sel2; };
    CacheOp cache_ops[CNT_CACHE];
    MathOp  math_ops[CNT_MATH];

    constexpr int max_ops = (CNT_CACHE > CNT_MATH) ? CNT_CACHE : CNT_MATH;

    for (int i = 0; i < max_ops; ++i) {
        if (i < (int)CNT_CACHE) {
            cache_ops[i] = { state.next_src(), state.next_dst(), state.rng() };
        }
        if (i < (int)CNT_MATH) {
            uint32_t sr = state.rng() % (NUM_REGS * (NUM_REGS - 1));
            uint32_t s1 = sr % NUM_REGS, s2 = sr / NUM_REGS;
            if (s2 >= s1) ++s2;
            uint32_t sel1 = state.rng();
            uint32_t mdst = state.next_dst();
            uint32_t sel2 = state.rng();
            math_ops[i] = { s1, s2, sel1, mdst, sel2 };
        }
    }

    // ---- Phase 1b: emit body (same pairing logic as generate_program) ----
    auto emit_math_block = [&](std::string& out, const MathOp& m) {
        std::string fused = try_fuse_math_merge(reg(m.dst), reg(m.src1), reg(m.src2), m.sel1, m.sel2);
        if (!fused.empty()) {
            out += "        " + fused + "\n";
        } else {
            out += "        _m = " + emit_math(reg(m.src1), reg(m.src2), m.sel1) + ";\n";
            out += "        " + emit_merge(reg(m.dst), "_m", m.sel2) + "\n";
        }
    };

    std::string body;
    body.reserve(6144);
    constexpr int MAX_K = 4;

    auto can_pair_k = [&](int i, int k) -> bool {
        if (i + k > (int)CNT_CACHE) return false;
        for (int j = 1; j < k; ++j) {
            uint32_t s = cache_ops[i + j].src;
            for (int p = 0; p < j; ++p) {
                if (s == cache_ops[i + p].dst) return false;
                if (i + p < (int)CNT_MATH && s == math_ops[i + p].dst) return false;
            }
        }
        return true;
    };

    body += "        uint32_t _c0, _c1, _c2, _c3, _m, _m1;\n";

    int i = 0;
    while (i < max_ops) {
        int k = 1;
        if (i < (int)CNT_CACHE) {
            int kmax = std::min(MAX_K, (int)CNT_CACHE - i);
            for (int t = kmax; t >= 2; --t) {
                if (can_pair_k(i, t)) { k = t; break; }
            }
        }

        if (k >= 2) {
            for (int j = 0; j < k; ++j) {
                body += "        _c" + std::to_string(j) + " = l1_cache["
                      + reg(cache_ops[i + j].src) + " & 0xFFFu];\n";
            }
            for (int j = 0; j < k; ++j) {
                body += "        " + emit_merge(reg(cache_ops[i + j].dst),
                                                "_c" + std::to_string(j),
                                                cache_ops[i + j].sel) + "\n";
                if (i + j < (int)CNT_MATH) emit_math_block(body, math_ops[i + j]);
            }
            i += k;
        } else if (i >= (int)CNT_CACHE && i + 1 < (int)CNT_MATH) {
            const auto& m0 = math_ops[i];
            const auto& m1 = math_ops[i + 1];
            bool can_pair = (m1.src1 != m0.dst && m1.src2 != m0.dst
                          && m0.src1 != m1.dst && m0.src2 != m1.dst
                          && m0.dst != m1.dst);
            if (can_pair) {
                std::string fused0 = try_fuse_math_merge(reg(m0.dst), reg(m0.src1), reg(m0.src2), m0.sel1, m0.sel2);
                std::string fused1 = try_fuse_math_merge(reg(m1.dst), reg(m1.src1), reg(m1.src2), m1.sel1, m1.sel2);
                if (fused0.empty() && fused1.empty()) {
                    body += "        _m = " + emit_math(reg(m0.src1), reg(m0.src2), m0.sel1) + ";\n";
                    body += "        _m1 = " + emit_math(reg(m1.src1), reg(m1.src2), m1.sel1) + ";\n";
                    body += "        " + emit_merge(reg(m0.dst), "_m", m0.sel2) + "\n";
                    body += "        " + emit_merge(reg(m1.dst), "_m1", m1.sel2) + "\n";
                    i += 2;
                } else {
                    emit_math_block(body, m0);
                    ++i;
                }
            } else {
                emit_math_block(body, m0);
                ++i;
            }
        } else {
            if (i < (int)CNT_CACHE) {
                body += "        _c0 = l1_cache[" + reg(cache_ops[i].src) + " & 0xFFFu];\n";
                body += "        " + emit_merge(reg(cache_ops[i].dst), "_c0", cache_ops[i].sel) + "\n";
            }
            if (i < (int)CNT_MATH) emit_math_block(body, math_ops[i]);
            ++i;
        }
    }

    // ---- Phase 2: generate DAG params (RNG advances AFTER cache+math) ----
    uint32_t dag_dsts[NUM_WORDS_PER_LANE];
    uint32_t dag_sels[NUM_WORDS_PER_LANE];
    for (uint32_t i = 0; i < NUM_WORDS_PER_LANE; ++i) {
        dag_dsts[i] = (i == 0) ? 0 : state.next_dst();
        dag_sels[i] = state.rng();
    }

    // ---- Phase 3: assemble — GLC asm load, then body, then merge ----
    std::string c;
    c.reserve(8192);
    c += buf; // header comment
    c += "\n";

    // GLC asm DAG load — volatile to pin scheduling
    c += "        const void* _p = d_dag + (dag_addr * 16u + ((lane_id ^ loop) & 15u)) * 4u;\n";
    c += "        uint4 _dg;\n";
    c += "#if __gfx11__ || __gfx1100__ || __gfx1101__ || __gfx1102__\n";
    c += "        asm volatile(\"global_load_b128 %0, %1, off glc\" : \"=v\"(_dg) : \"v\"(_p));\n";
    c += "#else\n";
    c += "        asm volatile(\"global_load_dwordx4 %0, %1, off glc\" : \"=v\"(_dg) : \"v\"(_p));\n";
    c += "#endif\n\n";

    // Cache + math body
    c += body;

    // DAG merge — split rot+xor so rotations execute while DAG load is in-flight
    c += "\n        // DAG merge (rotation-first split)\n";
    static const char* fields[] = {"_dg.x", "_dg.y", "_dg.z", "_dg.w"};
    // Phase 1: emit rotations into temps (no _dg dependency, can overlap with load)
    for (uint32_t i = 0; i < NUM_WORDS_PER_LANE; ++i) {
        uint32_t mtype = dag_sels[i] % 4;
        if (mtype == 2 || mtype == 3) {
            uint32_t x = ((dag_sels[i] >> 16) % 31) + 1;
            const char* rot = (mtype == 2) ? "rotl32" : "rotr32";
            c += "        uint32_t _r" + std::to_string(i) + " = " + rot + "(" + reg(dag_dsts[i]) + ", " + std::to_string(x) + "u);\n";
        }
    }
    // Phase 2: XOR with _dg (stalls on load if needed), then non-rot merges
    for (uint32_t i = 0; i < NUM_WORDS_PER_LANE; ++i) {
        uint32_t mtype = dag_sels[i] % 4;
        if (mtype == 2 || mtype == 3) {
            c += "        " + reg(dag_dsts[i]) + " = _r" + std::to_string(i) + " ^ " + fields[i] + ";\n";
        } else {
            c += "        " + emit_merge(reg(dag_dsts[i]), fields[i], dag_sels[i]) + "\n";
        }
    }

    return c;
}

// ---------------------------------------------------------------------------
// inject_coin_padding — replaces /* KAWPOW_COIN_PADDING */ with constexpr scalars
//
// Using individual constexpr instead of __constant__ array lets the RTC compiler
// embed each value as an inline immediate (s_mov_b32 imm) — zero constant-memory loads.
// ---------------------------------------------------------------------------
inline std::string inject_coin_padding(const std::string& kernel_source,
                                       const kawpow_coin_padding_t& padding) {
    std::string decls;
    char buf[80];
    for (int i = 0; i < 15; ++i) {
        snprintf(buf, sizeof(buf), "constexpr uint32_t KPAD%d = 0x%08xu;\n", i, padding.words[i]);
        decls += buf;
    }

    std::string result = kernel_source;
    const std::string marker = "/* KAWPOW_COIN_PADDING */";
    auto pos = result.find(marker);
    if (pos != std::string::npos)
        result.replace(pos, marker.size(), decls);
    return result;
}

// ---------------------------------------------------------------------------
// inject_dag_constants — replaces /* KAWPOW_DAG_CONSTS */ with compile-time defines
// ---------------------------------------------------------------------------
inline std::string inject_dag_constants(const std::string& kernel_source,
                                         uint32_t dag_num_items_div,
                                         uint32_t barrett_m,
                                         uint32_t barrett_shift) {
    char buf[256];
    snprintf(buf, sizeof(buf),
        "constexpr uint32_t KAWPOW_DAG_NUM_ITEMS_DIV = %uu;\n"
        "constexpr uint32_t KAWPOW_BARRETT_M         = 0x%08xu;\n"
        "constexpr uint32_t KAWPOW_BARRETT_SHIFT     = %uu;\n",
        dag_num_items_div, barrett_m, barrett_shift);

    std::string result = kernel_source;
    const std::string marker = "/* KAWPOW_DAG_CONSTS */";
    auto pos = result.find(marker);
    if (pos != std::string::npos)
        result.replace(pos, marker.size(), buf);
    return result;
}

// ---------------------------------------------------------------------------
// inject_program — replaces /* PROGPOW_PROGRAM */ in kernel source
// ---------------------------------------------------------------------------
inline std::string inject_program(const std::string& kernel_source, int block_number) {
    std::string result = kernel_source;
    const std::string marker = "/* PROGPOW_PROGRAM */";
    auto pos = result.find(marker);
    if (pos != std::string::npos) {
        result.replace(pos, marker.size(), generate_program(block_number));
    }
    const std::string glc_marker = "/* PROGPOW_PROGRAM_GLC */";
    pos = result.find(glc_marker);
    if (pos != std::string::npos) {
        result.replace(pos, glc_marker.size(), generate_program_glc(block_number));
    }
    return result;
}

} // namespace kawpow_proggen
