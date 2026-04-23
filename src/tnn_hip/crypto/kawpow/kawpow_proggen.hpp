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
#include <vector>

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
// Small bitmask helpers for dependency slicing
// ---------------------------------------------------------------------------
inline uint32_t bit(uint32_t r) {
    return 1u << r;
}

inline uint32_t mask1(uint32_t a) {
    return bit(a);
}

inline uint32_t mask2(uint32_t a, uint32_t b) {
    return bit(a) | bit(b);
}

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
// Body op + split containers
// ---------------------------------------------------------------------------
struct BodyOp {
    uint32_t reads_mask = 0;
    uint32_t writes_mask = 0;
    std::string code;
};

struct BodySplit {
    std::vector<BodyOp> pre;
    std::vector<BodyOp> post;
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

        uint32_t z     = fnv1a(FNV_OFFSET_BASIS, seed_lo);
        uint32_t w     = fnv1a(z, seed_hi);
        uint32_t jsr   = fnv1a(w, seed_lo);
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
// from the same operation sequence.
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
// Code emission helpers
// ---------------------------------------------------------------------------

// Parameterized register access: (mix)[r]
inline std::string reg(const std::string& mix, uint32_t r) {
    return "(" + mix + ")[" + std::to_string(r) + "]";
}

// Shorthand for expanded mode with "mix" array name
inline std::string reg(uint32_t r) {
    return "mix[" + std::to_string(r) + "]";
}

// Emit merge: a = f(a, b, sel)
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

// Helper: emit NVIDIA lop3.b32 inline asm, guarded by #ifndef __HIP_PLATFORM_AMD__
inline std::string emit_lop3(const std::string& dst,
                             const std::string& a,
                             const std::string& b,
                             const std::string& c,
                             uint32_t truth_table) {
    char tt[8];
    std::snprintf(tt, sizeof(tt), "0x%02x", truth_table);
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
    (void)dst; (void)src1; (void)src2; (void)math_sel; (void)merge_sel;
    return {};
}

// ---------------------------------------------------------------------------
// build_body_ops — generate structured cache+math body ops
//
// Important: temp producers (_c0/_c1/_m/_m1) are grouped with the merge that
// consumes them, so the dependency slicer cannot separate them illegally.
// ---------------------------------------------------------------------------
inline std::vector<BodyOp> build_body_ops(const ProgramOps& ops, const std::string& mix) {
    using MathOp = ProgramOps::MathOp;
    std::vector<BodyOp> out;
    out.reserve(CNT_CACHE + CNT_MATH + 8);

    auto r = [&](uint32_t idx) -> std::string { return reg(mix, idx); };

    auto push_raw = [&](uint32_t reads, uint32_t writes, std::string code) {
        out.push_back({reads, writes, std::move(code)});
    };

    auto emit_math_block = [&](const MathOp& m) {
        std::string fused = try_fuse_math_merge(r(m.dst), r(m.src1), r(m.src2), m.sel1, m.sel2);
        if (!fused.empty()) {
            push_raw(mask1(m.dst) | mask2(m.src1, m.src2), mask1(m.dst),
                     "    " + fused + ";");
        } else {
            std::string code;
            code += "    _m = " + emit_math(r(m.src1), r(m.src2), m.sel1) + ";\n";
            code += "    " + emit_merge(r(m.dst), "_m", m.sel2);
            push_raw(mask1(m.dst) | mask2(m.src1, m.src2), mask1(m.dst), std::move(code));
        }
    };

    constexpr int MAX_K = 2;
    constexpr int max_ops = (CNT_CACHE > CNT_MATH) ? CNT_CACHE : CNT_MATH;

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

    // temp decl, no deps
    push_raw(0, 0, "    uint32_t _c0, _c1, _m, _m1;");

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
                uint32_t src = ops.cache_ops[i + j].src;
                uint32_t dst = ops.cache_ops[i + j].dst;

                std::string code;
                code += "    _c" + std::to_string(j) + " = l1_cache[" + r(src) + " & 0xFFFu];\n";
                code += "    " + emit_merge(r(dst), "_c" + std::to_string(j), ops.cache_ops[i + j].sel);

                push_raw(mask1(src) | mask1(dst), mask1(dst), std::move(code));

                if (i + j < (int)CNT_MATH) emit_math_block(ops.math_ops[i + j]);
            }
            i += k;
        } else if (i >= (int)CNT_CACHE && i + 1 < (int)CNT_MATH) {
            const auto& m0 = ops.math_ops[i];
            const auto& m1 = ops.math_ops[i + 1];
            bool can_pair = (m1.src1 != m0.dst && m1.src2 != m0.dst
                          && m0.src1 != m1.dst && m0.src2 != m1.dst
                          && m0.dst != m1.dst);
            if (can_pair) {
                std::string code0;
                code0 += "    _m = " + emit_math(r(m0.src1), r(m0.src2), m0.sel1) + ";\n";
                code0 += "    " + emit_merge(r(m0.dst), "_m", m0.sel2);
                push_raw(mask1(m0.dst) | mask2(m0.src1, m0.src2), mask1(m0.dst), std::move(code0));

                std::string code1;
                code1 += "    _m1 = " + emit_math(r(m1.src1), r(m1.src2), m1.sel1) + ";\n";
                code1 += "    " + emit_merge(r(m1.dst), "_m1", m1.sel2);
                push_raw(mask1(m1.dst) | mask2(m1.src1, m1.src2), mask1(m1.dst), std::move(code1));

                i += 2;
            } else {
                emit_math_block(m0);
                ++i;
            }
        } else {
            if (i < (int)CNT_CACHE) {
                uint32_t src = ops.cache_ops[i].src;
                uint32_t dst = ops.cache_ops[i].dst;

                std::string code;
                code += "    _c0 = l1_cache[" + r(src) + " & 0xFFFu];\n";
                code += "    " + emit_merge(r(dst), "_c0", ops.cache_ops[i].sel);

                push_raw(mask1(src) | mask1(dst), mask1(dst), std::move(code));
            }
            if (i < (int)CNT_MATH) emit_math_block(ops.math_ops[i]);
            ++i;
        }
    }

    return out;
}

inline BodySplit split_body_at_last_write_to(const std::vector<BodyOp>& ops, uint32_t reg_idx) {
    BodySplit s;
    if (ops.empty()) return s;

    const uint32_t target = bit(reg_idx);
    int cut = -1;

    for (int i = 0; i < (int)ops.size(); ++i) {
        if (ops[i].writes_mask & target) {
            cut = i;
        }
    }

    // No write to the target register in body: no useful split
    if (cut < 0) {
        s.pre = ops;
        return s;
    }

    s.pre.insert(s.pre.end(), ops.begin(), ops.begin() + cut + 1);
    s.post.insert(s.post.end(), ops.begin() + cut + 1, ops.end());
    return s;
}

// Emit ops into either expanded text or macro text.
// Important: embedded newlines inside BodyOp.code must also receive the
// requested line ending, or macro emission will break.
inline std::string emit_body_ops(const std::vector<BodyOp>& ops, const std::string& le) {
    std::string body;
    body.reserve(4096);

    for (const auto& op : ops) {
        for (char ch : op.code) {
            if (ch == '\n') body += le;
            else            body += ch;
        }
        body += le;
    }

    return body;
}

// ---------------------------------------------------------------------------
// emit_body — full cache+math body
// ---------------------------------------------------------------------------
inline std::string emit_body(const ProgramOps& ops, const std::string& mix,
                             const std::string& le) {
    return emit_body_ops(build_body_ops(ops, mix), le);
}

// ---------------------------------------------------------------------------
// DAG helpers
// ---------------------------------------------------------------------------

// Full capture, mainly for experimentation / dump helpers
inline std::string emit_dag_capture(const std::string& dg, const std::string& le) {
    std::string c;
    c += "    uint32_t _w0 = dag_word0(" + dg + ");" + le;
    c += "    uint32_t _w1 = dag_word1(" + dg + ");" + le;
    c += "    uint32_t _w2 = dag_word2(" + dg + ");" + le;
    c += "    uint32_t _w3 = dag_word3(" + dg + ");" + le;
    return c;
}

// Compute the future merge[0] result into a temp for early ISSUE_DAG_FROM.
// This must NOT write MIX[0] yet, because later body ops may still read/write it.
inline std::string emit_dag_prepare_issue_value(const ProgramOps& ops,
                                                const std::string& mix,
                                                const std::string& dg,
                                                const std::string& le) {
    auto r = [&](uint32_t idx) -> std::string { return reg(mix, idx); };

    std::string c;
    c += "    uint32_t _w0 = dag_word0(" + dg + ");" + le;

    // In collect_ops(), dag_dsts[0] is forced to 0, so this is MIX[0]-based.
    const std::string a = r(ops.dag_dsts[0]);
    uint32_t sel = ops.dag_sels[0];
    uint32_t mtype0 = sel % 4;

    if (mtype0 == 2 || mtype0 == 3) {
        uint32_t x = ((sel >> 16) % 31) + 1;
        const char* rot = (mtype0 == 2) ? "rotl32" : "rotr32";
        c += "    uint32_t _mix0_next = " + std::string(rot) + "("
          + a + ", " + std::to_string(x) + "u) ^ _w0;" + le;
    } else if (mtype0 == 0) {
        c += "    uint32_t _mix0_next = (" + a + " * 33u) + _w0;" + le;
    } else {
        c += "    uint32_t _mix0_next = (" + a + " ^ _w0) * 33u;" + le;
    }

    return c;
}

// Capture remaining DAG words after issue
inline std::string emit_dag_capture_post(const std::string& dg,
                                         const std::string& le) {
    std::string c;
    c += "    uint32_t _w1 = dag_word1(" + dg + ");" + le;
    c += "    uint32_t _w2 = dag_word2(" + dg + ");" + le;
    c += "    uint32_t _w3 = dag_word3(" + dg + ");" + le;
    return c;
}

// Commit the delayed merge[0] result to MIX[0] at the original semantic point
inline std::string emit_dag_commit_pre(const ProgramOps& ops,
                                       const std::string& mix,
                                       const std::string& le) {
    auto r = [&](uint32_t idx) -> std::string { return reg(mix, idx); };
    return "    " + r(ops.dag_dsts[0]) + " = _mix0_next;" + le;
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
// generate_program_macro — emit body macros
// ---------------------------------------------------------------------------
inline std::string generate_program_macro(int block_number) {
    auto ops = collect_ops(block_number);
    auto body_ops = build_body_ops(ops, "MIX");
    auto split = split_body_at_last_write_to(body_ops, 0);  // early address depends on MIX[0]

    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "// ProgPoW body macro for period %llu (block %d)\n",
        (unsigned long long)ops.period, block_number);

    std::string c;
    c.reserve(8192);
    c += buf;

    c += "#define PROGPOW_BODY(MIX, DG) do { \\\n";
    c += emit_body_ops(body_ops, " \\\n");
    c += emit_dag_merge(ops, "MIX", "DG", " \\\n");
    c += "} while(0)\n";
    c += "\n";

    c += "#define PROGPOW_BODY_PIPE(MIX, DG_CUR, DG_NEXT, NEXT_LOOP) do { \\\n";
    c += emit_body_ops(split.pre, " \\\n");
    c += emit_dag_prepare_issue_value(ops, "MIX", "DG_CUR", " \\\n");
    c += "    PROGPOW_ISSUE_DAG_FROM(_mix0_next, DG_NEXT, NEXT_LOOP); \\\n";
    c += emit_body_ops(split.post, " \\\n");
    c += emit_dag_capture_post("DG_CUR", " \\\n");
    c += emit_dag_commit_pre(ops, "MIX", " \\\n");
    c += emit_dag_merge_post(ops, "MIX", " \\\n");
    c += "} while(0)\n";
    c += "\n";

    return c;
}

// ---------------------------------------------------------------------------
// generate_program_expanded — expanded body for dump/diagnostic purposes
// ---------------------------------------------------------------------------
inline std::string generate_program_expanded(int block_number) {
    auto ops = collect_ops(block_number);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "    // ProgPoW program for period %llu (block %d)\n",
        (unsigned long long)ops.period, block_number);

    std::string c;
    c.reserve(8192);
    c += buf;
    c += "\n";
    c += emit_body(ops, "mix", "\n");
    c += "\n    // DAG merge (direct)\n";
    c += emit_dag_merge(ops, "mix", "_dg", "\n");
    return c;
}

// Backwards-compatible alias for dump/test tools
inline std::string generate_program(int block_number) {
    return generate_program_expanded(block_number);
}

// ---------------------------------------------------------------------------
// inject_coin_padding
// ---------------------------------------------------------------------------
inline std::string inject_coin_padding(const std::string& kernel_source,
                                       const kawpow_coin_padding_t& padding) {
    std::string decls;
    char buf[80];
    for (int i = 0; i < 15; ++i) {
        std::snprintf(buf, sizeof(buf), "constexpr uint32_t KPAD%d = 0x%08xu;\n", i, padding.words[i]);
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
// inject_dag_constants
// ---------------------------------------------------------------------------
inline std::string inject_dag_constants(const std::string& kernel_source,
                                        uint32_t dag_num_items_div,
                                        uint32_t barrett_m,
                                        uint32_t barrett_shift) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
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
// inject_program
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