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
// Fused math+merge for compound instruction patterns only.
// Returns empty string if no fusion applies (caller falls back to _m temp).
//
// IMPORTANT: never strength-reduce * 33u here — the compiler does it better.
// Only fuse when the combined expression exposes a three-operand ISA pattern
// that the _m temp would obscure.
//
// v_add3_u32 patterns (three-way add):
//   math(add:0)       + merge(mul33+add:0): (dst*33) + a + b
//   math(clz+clz:9)   + merge(mul33+add:0): (dst*33) + __clz(a) + __clz(b)
//   math(popc+popc:10) + merge(mul33+add:0): (dst*33) + __popc(a) + __popc(b)
//
// v_xor3_b32 patterns (three-way xor):
//   math(xor:8) + merge(rotl^:2):    rotl32(dst, x) ^ a ^ b
//   math(xor:8) + merge(rotr^:3):    rotr32(dst, x) ^ a ^ b
//   math(xor:8) + merge(xor_mul33:1): (dst ^ a ^ b) * 33
// ---------------------------------------------------------------------------
inline std::string try_fuse_math_merge(const std::string& dst,
                                       const std::string& src1,
                                       const std::string& src2,
                                       uint32_t math_sel,
                                       uint32_t merge_sel) {
    uint32_t math_op  = math_sel % 11;
    uint32_t merge_op = merge_sel % 4;

    // ---- v_add3 patterns: merge(0) with add-like math ----
    if (merge_op == 0) {
        // add + mul33_add: (dst * 33) + src1 + src2
        if (math_op == 0) {
            return dst + " = (" + dst + " * 33u) + " + src1 + " + " + src2 + ";";
        }
        // clz+clz + mul33_add: (dst * 33) + __clz(src1) + __clz(src2)
        if (math_op == 9) {
            return dst + " = (" + dst + " * 33u) + __clz(" + src1 + ") + __clz(" + src2 + ");";
        }
        // popc+popc + mul33_add: (dst * 33) + __popc(src1) + __popc(src2)
        if (math_op == 10) {
            return dst + " = (" + dst + " * 33u) + __popc(" + src1 + ") + __popc(" + src2 + ");";
        }
    }

    // ---- v_xor3 patterns: math(xor:8) with xor-like merge ----
    if (math_op == 8) {
        // xor + xor_mul33: (dst ^ src1 ^ src2) * 33
        if (merge_op == 1) {
            return dst + " = (" + dst + " ^ " + src1 + " ^ " + src2 + ") * 33u;";
        }
        // xor + rotl_xor: rotl32(dst, x) ^ src1 ^ src2
        if (merge_op == 2) {
            uint32_t x = ((merge_sel >> 16) % 31) + 1;
            return dst + " = rotl32(" + dst + ", " + std::to_string(x) + "u) ^ " + src1 + " ^ " + src2 + ";";
        }
        // xor + rotr_xor: rotr32(dst, x) ^ src1 ^ src2
        if (merge_op == 3) {
            uint32_t x = ((merge_sel >> 16) % 31) + 1;
            return dst + " = rotr32(" + dst + ", " + std::to_string(x) + "u) ^ " + src1 + " ^ " + src2 + ";";
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

    // ---- Phase 1: generate cache+math code (advances RNG in spec order) ----
    std::string body;
    body.reserve(6144);

    // Shared temps — one declaration, reused across all rounds
    body += "        uint32_t _c, _m;\n";

    constexpr int max_ops = (CNT_CACHE > CNT_MATH) ? CNT_CACHE : CNT_MATH;

    for (int i = 0; i < max_ops; ++i) {

        if (i < (int)CNT_CACHE) {
            uint32_t src = state.next_src();
            uint32_t dst = state.next_dst();
            uint32_t sel = state.rng();

            // AND mask instead of modulo (4096 = 2^12)
            body += "        _c = l1_cache[" + reg(src) + " & 0xFFFu];\n";
            body += "        " + emit_merge(reg(dst), "_c", sel) + "\n";
        }

        if (i < (int)CNT_MATH) {
            uint32_t src_rnd = state.rng() % (NUM_REGS * (NUM_REGS - 1));
            uint32_t src1 = src_rnd % NUM_REGS;
            uint32_t src2 = src_rnd / NUM_REGS;
            if (src2 >= src1) ++src2;

            uint32_t sel1 = state.rng();
            uint32_t dst  = state.next_dst();
            uint32_t sel2 = state.rng();

            std::string fused = try_fuse_math_merge(reg(dst), reg(src1), reg(src2), sel1, sel2);
            if (!fused.empty()) {
                body += "        " + fused + "\n";
            } else {
                body += "        _m = " + emit_math(reg(src1), reg(src2), sel1) + ";\n";
                body += "        " + emit_merge(reg(dst), "_m", sel2) + "\n";
            }
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
    // Single uint4 load → compiler emits global_load_b128 (vectorized, not 4 scalar loads)
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
// inject_coin_padding — replaces /* KAWPOW_COIN_PADDING */ with a constant array
// ---------------------------------------------------------------------------
inline std::string inject_coin_padding(const std::string& kernel_source,
                                       const kawpow_coin_padding_t& padding) {
    std::string arr = "__device__ __constant__ static const uint32_t kawpow_coin_padding[15] = {\n";
    char buf[64];
    for (int i = 0; i < 15; ++i) {
        snprintf(buf, sizeof(buf), "    0x%08xu%s\n", padding.words[i], i < 14 ? "," : "");
        arr += buf;
    }
    arr += "};\n";

    std::string result = kernel_source;
    const std::string marker = "/* KAWPOW_COIN_PADDING */";
    auto pos = result.find(marker);
    if (pos != std::string::npos)
        result.replace(pos, marker.size(), arr);
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
    return result;
}

} // namespace kawpow_proggen
