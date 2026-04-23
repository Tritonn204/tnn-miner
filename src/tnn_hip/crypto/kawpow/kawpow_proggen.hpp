#pragma once
// ============================================================================
// kawpow_proggen.hpp — KAWPOW random-program JIT source generator
//
// Precisely mirrors the RNG sequence from cpp-kawpow's progpow.cpp round().
// The generated code is injected into kawpow.hip at /* PROGPOW_MACROS */
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
// ProgramOps — collected RNG-derived operations for a given period.
// Separates RNG replay from code emission, enabling multiple output formats
// (macro, expanded, GLC) from the same operation sequence.
// ---------------------------------------------------------------------------
struct ProgramOps {
    struct CacheOp { uint32_t src, dst, sel; };
    struct MathOp  { uint32_t src1, src2, sel1, dst, sel2; };

    CacheOp  cache_ops[CNT_CACHE];
    MathOp   math_ops[CNT_MATH];
    uint32_t dag_dsts[NUM_WORDS_PER_LANE];
    uint32_t dag_sels[NUM_WORDS_PER_LANE];
    uint64_t period;
    int      block_number;
};

inline ProgramOps collect_ops(int block_number) {
    ProgramOps ops;
    ops.block_number = block_number;
    ops.period = static_cast<uint64_t>(block_number) / 3;
    MixRngState state(ops.period);

    constexpr int max_ops = (CNT_CACHE > CNT_MATH) ? CNT_CACHE : CNT_MATH;

    for (int i = 0; i < max_ops; ++i) {
        if (i < (int)CNT_CACHE) {
            ops.cache_ops[i] = { state.next_src(), state.next_dst(), state.rng() };
        }
        if (i < (int)CNT_MATH) {
            uint32_t sr = state.rng() % (NUM_REGS * (NUM_REGS - 1));
            uint32_t s1 = sr % NUM_REGS, s2 = sr / NUM_REGS;
            if (s2 >= s1) ++s2;
            uint32_t sel1 = state.rng();
            uint32_t mdst = state.next_dst();
            uint32_t sel2 = state.rng();
            ops.math_ops[i] = { s1, s2, sel1, mdst, sel2 };
        }
    }

    for (uint32_t i = 0; i < NUM_WORDS_PER_LANE; ++i) {
        ops.dag_dsts[i] = (i == 0) ? 0 : state.next_dst();
        ops.dag_sels[i] = state.rng();
    }

    return ops;
}

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

// Parameterized register access: (mix)[r]
inline std::string reg(const std::string& mix, uint32_t r) {
    return "(" + mix + ")[" + std::to_string(r) + "]";
}

// Shorthand for expanded mode with "mix" array name
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
    // No fusion — always use _m temp, let the compiler decide
    return {};
}

// ---------------------------------------------------------------------------
// emit_body — generate the cache+math body for a ProgPoW iteration.
//
// Parameters:
//   ops  — collected RNG-derived operations
//   mix  — name of the mix array (e.g. "MIX" for macro, "mix" for expanded)
//   le   — line ending ("\n" for expanded, " \\\n" for macro continuation)
// ---------------------------------------------------------------------------
inline std::string emit_body(const ProgramOps& ops, const std::string& mix,
                              const std::string& le) {
    using MathOp = ProgramOps::MathOp;

    auto r = [&](uint32_t idx) -> std::string { return reg(mix, idx); };

    auto emit_math_block = [&](std::string& out, const MathOp& m) {
        std::string fused = try_fuse_math_merge(r(m.dst), r(m.src1), r(m.src2), m.sel1, m.sel2);
        if (!fused.empty()) {
            out += "    " + fused + le;
        } else {
            out += "    _m = " + emit_math(r(m.src1), r(m.src2), m.sel1) + ";" + le;
            out += "    " + emit_merge(r(m.dst), "_m", m.sel2) + le;
        }
    };

    std::string body;
    body.reserve(6144);
    constexpr int MAX_K = 2;
    constexpr int max_ops = (CNT_CACHE > CNT_MATH) ? CNT_CACHE : CNT_MATH;

    // Check if k consecutive cache reads are safe to issue back-to-back.
    // Safe when: cache[i+j].src is NOT written by any prior cache[i+p].merge or math[i+p].
    auto can_pair_k = [&](int i, int k) -> bool {
        if (i + k > (int)CNT_CACHE) return false;
        for (int j = 1; j < k; ++j) {
            uint32_t s = ops.cache_ops[i + j].src;
            for (int p = 0; p < j; ++p) {
                if (s == ops.cache_ops[i + p].dst) return false;
                if (i + p < (int)CNT_MATH && s == ops.math_ops[i + p].dst) return false;
            }
        }
        return true;
    };

    body += "    uint32_t _c0, _c1, _m, _m1;" + le;

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
            // Issue k LDS reads up front
            for (int j = 0; j < k; ++j) {
                body += "    _c" + std::to_string(j) + " = l1_cache["
                      + r(ops.cache_ops[i + j].src) + " & 0xFFFu];" + le;
            }
            // Interleave merges and maths in original order
            for (int j = 0; j < k; ++j) {
                body += "    " + emit_merge(r(ops.cache_ops[i + j].dst),
                                            "_c" + std::to_string(j),
                                            ops.cache_ops[i + j].sel) + le;
                if (i + j < (int)CNT_MATH) emit_math_block(body, ops.math_ops[i + j]);
            }
            i += k;
        } else if (i >= (int)CNT_CACHE && i + 1 < (int)CNT_MATH) {
            // Tail: pure math ops — pair adjacent ops for VOPD packing
            const auto& m0 = ops.math_ops[i];
            const auto& m1 = ops.math_ops[i + 1];
            bool can_pair = (m1.src1 != m0.dst && m1.src2 != m0.dst
                          && m0.src1 != m1.dst && m0.src2 != m1.dst
                          && m0.dst != m1.dst);
            if (can_pair) {
                body += "    _m = " + emit_math(r(m0.src1), r(m0.src2), m0.sel1) + ";" + le;
                body += "    _m1 = " + emit_math(r(m1.src1), r(m1.src2), m1.sel1) + ";" + le;
                body += "    " + emit_merge(r(m0.dst), "_m", m0.sel2) + le;
                body += "    " + emit_merge(r(m1.dst), "_m1", m1.sel2) + le;
                i += 2;
            } else {
                emit_math_block(body, m0);
                ++i;
            }
        } else {
            // 1-way fallback
            if (i < (int)CNT_CACHE) {
                body += "    _c0 = l1_cache[" + r(ops.cache_ops[i].src) + " & 0xFFFu];" + le;
                body += "    " + emit_merge(r(ops.cache_ops[i].dst), "_c0", ops.cache_ops[i].sel) + le;
            }
            if (i < (int)CNT_MATH) emit_math_block(body, ops.math_ops[i]);
            ++i;
        }
    }

    return body;
}

inline std::string emit_dag_capture(const std::string& dg, const std::string& le) {
    std::string c;
    c += "    uint32_t _w0 = dag_word0(" + dg + ");" + le;
    c += "    uint32_t _w1 = dag_word1(" + dg + ");" + le;
    c += "    uint32_t _w2 = dag_word2(" + dg + ");" + le;
    c += "    uint32_t _w3 = dag_word3(" + dg + ");" + le;
    return c;
}

// ---------------------------------------------------------------------------
// emit_dag_merge_pre — merge[0] only (finalizes mix[0] for ISSUE_DAG).
// Only _r0 is pre-computed here; _r1/_r2/_r3 are deferred to post so
// they execute after ISSUE_DAG, giving the new load more shadow and
// freeing 3 VGPRs of live range across the bpermute/Barrett region.
// ---------------------------------------------------------------------------
inline std::string emit_dag_capture_pre(const ProgramOps& ops,
                                        const std::string& mix,
                                        const std::string& dg,
                                        const std::string& le) {
    auto r = [&](uint32_t idx) -> std::string { return reg(mix, idx); };

    std::string c;

    // Capture word 0 only
    c += "    uint32_t _w0 = dag_word0(" + dg + ");" + le;

    // Apply merge[0] immediately because it feeds next DAG address via MIX[0]
    uint32_t mtype0 = ops.dag_sels[0] % 4;
    if (mtype0 == 2 || mtype0 == 3) {
        uint32_t x = ((ops.dag_sels[0] >> 16) % 31) + 1;
        const char* rot = (mtype0 == 2) ? "rotl32" : "rotr32";
        c += "    " + r(ops.dag_dsts[0]) + " = " + rot + "("
          + r(ops.dag_dsts[0]) + ", " + std::to_string(x) + "u) ^ _w0;" + le;
    } else {
        c += "    " + emit_merge(r(ops.dag_dsts[0]), "_w0", ops.dag_sels[0]) + le;
    }

    return c;
}

// ---------------------------------------------------------------------------
// emit_dag_merge_post — merge[1..3] with inline rotation (no temps).
// Rotations are computed inline with the XOR — executes after ISSUE_DAG,
// giving the new load shadow cycles from the rot+xor instructions.
// ---------------------------------------------------------------------------
inline std::string emit_dag_capture_post(const std::string& dg,
                                         const std::string& le) {
    std::string c;
    c += "    uint32_t _w1 = dag_word1(" + dg + ");" + le;
    c += "    uint32_t _w2 = dag_word2(" + dg + ");" + le;
    c += "    uint32_t _w3 = dag_word3(" + dg + ");" + le;
    return c;
}

inline std::string emit_dag_merge_post(const ProgramOps& ops,
                                       const std::string& mix,
                                       const std::string& le) {
    auto r = [&](uint32_t idx) -> std::string { return reg(mix, idx); };

    std::string c;

    for (uint32_t i = 1; i < NUM_WORDS_PER_LANE; ++i) {
        std::string field = "_w" + std::to_string(i);
        uint32_t mtype = ops.dag_sels[i] % 4;

        if (mtype == 2 || mtype == 3) {
            uint32_t x = ((ops.dag_sels[i] >> 16) % 31) + 1;
            const char* rot = (mtype == 2) ? "rotl32" : "rotr32";
            c += "    " + r(ops.dag_dsts[i]) + " = " + rot + "("
              + r(ops.dag_dsts[i]) + ", " + std::to_string(x) + "u) ^ " + field + ";" + le;
        } else {
            c += "    " + emit_merge(r(ops.dag_dsts[i]), field, ops.dag_sels[i]) + le;
        }
    }

    return c;
}

// Direct versions for full non-pipelined body
inline std::string emit_dag_merge_pre_direct(const ProgramOps& ops,
                                             const std::string& mix,
                                             const std::string& dg,
                                             const std::string& le) {
    auto r = [&](uint32_t idx) -> std::string { return reg(mix, idx); };

    std::string field0 = "dag_word0(" + dg + ")";
    std::string c;

    uint32_t mtype0 = ops.dag_sels[0] % 4;
    if (mtype0 == 2 || mtype0 == 3) {
        uint32_t x = ((ops.dag_sels[0] >> 16) % 31) + 1;
        const char* rot = (mtype0 == 2) ? "rotl32" : "rotr32";
        c += "    " + r(ops.dag_dsts[0]) + " = " + rot + "("
          + r(ops.dag_dsts[0]) + ", " + std::to_string(x) + "u) ^ " + field0 + ";" + le;
    } else {
        c += "    " + emit_merge(r(ops.dag_dsts[0]), field0, ops.dag_sels[0]) + le;
    }

    return c;
}

inline std::string emit_dag_merge_post_direct(const ProgramOps& ops,
                                              const std::string& mix,
                                              const std::string& dg,
                                              const std::string& le) {
    auto r = [&](uint32_t idx) -> std::string { return reg(mix, idx); };

    std::string c;

    for (uint32_t i = 1; i < NUM_WORDS_PER_LANE; ++i) {
        std::string field = "dag_word" + std::to_string(i) + "(" + dg + ")";
        uint32_t mtype = ops.dag_sels[i] % 4;
        if (mtype == 2 || mtype == 3) {
            uint32_t x = ((ops.dag_sels[i] >> 16) % 31) + 1;
            const char* rot = (mtype == 2) ? "rotl32" : "rotr32";
            c += "    " + r(ops.dag_dsts[i]) + " = " + rot + "("
              + r(ops.dag_dsts[i]) + ", " + std::to_string(x) + "u) ^ " + field + ";" + le;
        } else {
            c += "    " + emit_merge(r(ops.dag_dsts[i]), field, ops.dag_sels[i]) + le;
        }
    }

    return c;
}

inline std::string emit_dag_merge(const ProgramOps& ops,
                                  const std::string& mix,
                                  const std::string& dg,
                                  const std::string& le) {
    return emit_dag_merge_pre_direct(ops, mix, dg, le) +
           emit_dag_merge_post_direct(ops, mix, dg, le);
}

// ---------------------------------------------------------------------------
// generate_program_macro — emit #define PROGPOW_BODY(MIX, DG) macro
//
// The macro takes:
//   MIX — name of the uint32_t[32] mix array
//   DG  — name of a dag128_u64 variable holding pre-loaded DAG data
//
// All temporaries (_c0, _c1, _m, _m1) are scoped inside do{}while(0).
// The macro references l1_cache from the enclosing function scope.
// ---------------------------------------------------------------------------
inline std::string generate_program_macro(int block_number) {
    auto ops = collect_ops(block_number);
    char buf[256];
    snprintf(buf, sizeof(buf),
        "// ProgPoW body macro for period %llu (block %d)\n",
        (unsigned long long)ops.period, block_number);

    std::string c;
    c.reserve(8192);
    c += buf;

    c += "#define PROGPOW_BODY(MIX, DG) do { \\\n";
    c += emit_body(ops, "MIX", " \\\n");
    c += emit_dag_merge(ops, "MIX", "DG", " \\\n");
    c += "} while(0)\n";
    c += "\n";

    c += "#define PROGPOW_BODY_PIPE(MIX, DG_CUR, DG_NEXT, NEXT_LOOP) do { \\\n";
    c += emit_body(ops, "MIX", " \\\n");
    c += emit_dag_capture_pre(ops, "MIX", "DG_CUR", " \\\n");
    c += emit_dag_capture_post("DG_CUR", " \\\n");
    c += "    PROGPOW_ISSUE_DAG(MIX, DG_NEXT, NEXT_LOOP); \\\n";
    c += emit_dag_merge_post(ops, "MIX", " \\\n");
    c += "} while(0)\n";
    c += "\n";

    return c;
}

// ---------------------------------------------------------------------------
// generate_program_expanded — expanded body for dump/diagnostic purposes.
// Uses "mix" and "_dg" as variable names (not macro parameters).
// ---------------------------------------------------------------------------
inline std::string generate_program_expanded(int block_number) {
    auto ops = collect_ops(block_number);
    char buf[256];
    snprintf(buf, sizeof(buf),
        "    // ProgPoW program for period %llu (block %d)\n",
        (unsigned long long)ops.period, block_number);

    std::string c;
    c.reserve(8192);
    c += buf;
    c += "\n";
    c += emit_body(ops, "mix", "\n");
    c += "\n    // DAG merge (rotation-first split)\n";
    c += emit_dag_merge(ops, "mix", "_dg", "\n");
    return c;
}

// Backwards-compatible alias for dump/test tools
inline std::string generate_program(int block_number) {
    return generate_program_expanded(block_number);
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
// inject_program — replaces /* PROGPOW_MACROS */ with generated body macros
// ---------------------------------------------------------------------------
inline std::string inject_program(const std::string& kernel_source, int block_number) {
    std::string result = kernel_source;

    const std::string macro_marker = "/* PROGPOW_MACROS */";
    auto pos = result.find(macro_marker);
    if (pos != std::string::npos)
        result.replace(pos, macro_marker.size(), generate_program_macro(block_number));

    return result;
}

} // namespace kawpow_proggen
