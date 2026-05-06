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
// Body IR — typed representation of generated cache/math operations.
//
// This sits between RNG replay and string emission. It allows us to emit
// fused/unfused variants and to run dependency-aware split/scheduling passes
// without manipulating source strings directly.
// ---------------------------------------------------------------------------
enum class NodeKind {
    CacheMerge,
    MathMerge
};

struct BodyNode {
    NodeKind kind = NodeKind::MathMerge;
    uint32_t reads_mask = 0;
    uint32_t writes_mask = 0;

    // common
    uint32_t dst = 0;

    // CacheMerge: dst = merge(dst, l1_cache[src & 0xfff], cache_sel)
    uint32_t cache_src = 0;
    uint32_t cache_sel = 0;

    // MathMerge: tmp = math(src1, src2, math_sel); dst = merge(dst, tmp, merge_sel)
    uint32_t src1 = 0;
    uint32_t src2 = 0;
    uint32_t math_sel = 0;
    uint32_t merge_sel = 0;
};

struct BodyTriSplit {
    std::vector<BodyOp> critical;
    std::vector<BodyOp> filler;
    std::vector<BodyOp> post;
};

inline bool ops_conflict(const BodyOp& a, const BodyOp& b) {
    return ((a.writes_mask & (b.reads_mask | b.writes_mask)) != 0) ||
           ((b.writes_mask & (a.reads_mask | a.writes_mask)) != 0);
}

// ---------------------------------------------------------------------------
// build_body_nodes — RNG-order IR nodes for cache+math body.
//
// Important: each node preserves the original coupled operation semantics:
// cache/math producer plus its merge consumer remain one unit. Later passes can
// move whole nodes, but cannot split a temp producer away from its merge.
// ---------------------------------------------------------------------------
inline std::vector<BodyNode> build_body_nodes(const ProgramOps& ops) {
    std::vector<BodyNode> out;
    out.reserve(CNT_CACHE + CNT_MATH);

    constexpr int max_ops = (CNT_CACHE > CNT_MATH) ? CNT_CACHE : CNT_MATH;

    for (int i = 0; i < max_ops; ++i) {
        if (i < (int)CNT_CACHE) {
            const auto& c = ops.cache_ops[i];
            BodyNode n;
            n.kind = NodeKind::CacheMerge;
            n.cache_src = c.src;
            n.dst = c.dst;
            n.cache_sel = c.sel;
            n.reads_mask = mask1(c.src) | mask1(c.dst);
            n.writes_mask = mask1(c.dst);
            out.push_back(n);
        }

        if (i < (int)CNT_MATH) {
            const auto& m = ops.math_ops[i];
            BodyNode n;
            n.kind = NodeKind::MathMerge;
            n.src1 = m.src1;
            n.src2 = m.src2;
            n.dst = m.dst;
            n.math_sel = m.sel1;
            n.merge_sel = m.sel2;
            n.reads_mask = mask1(m.dst) | mask2(m.src1, m.src2);
            n.writes_mask = mask1(m.dst);
            out.push_back(n);
        }
    }

    return out;
}

template<bool ENABLE_FUSION = false>
inline BodyOp emit_node_as_bodyop(const BodyNode& n, const std::string& mix) {
    auto r = [&](uint32_t idx) -> std::string { return reg(mix, idx); };

    BodyOp op;
    op.reads_mask = n.reads_mask;
    op.writes_mask = n.writes_mask;

    if (n.kind == NodeKind::CacheMerge) {
        std::string code;
        code += "    _c0 = l1_cache[" + r(n.cache_src) + " & 0xFFFu];\n";
        code += "    " + emit_merge(r(n.dst), "_c0", n.cache_sel);
        op.code = std::move(code);
        return op;
    }

    if constexpr (ENABLE_FUSION) {
        std::string fused = try_fuse_math_merge(
            r(n.dst), r(n.src1), r(n.src2), n.math_sel, n.merge_sel);

        if (!fused.empty()) {
            op.code = "    " + fused + ";";
            return op;
        }
    }

    std::string code;
    code += "    _m = " + emit_math(r(n.src1), r(n.src2), n.math_sel) + ";\n";
    code += "    " + emit_merge(r(n.dst), "_m", n.merge_sel);
    op.code = std::move(code);
    return op;
}

template<bool ENABLE_FUSION = false>
inline std::vector<BodyOp> emit_body_ops_from_nodes(
    const std::vector<BodyNode>& nodes,
    const std::string& mix)
{
    std::vector<BodyOp> out;
    out.reserve(nodes.size());

    for (const auto& n : nodes)
        out.push_back(emit_node_as_bodyop<ENABLE_FUSION>(n, mix));

    return out;
}

template<bool ENABLE_FUSION = false>
inline std::vector<BodyOp> build_body_ops(const ProgramOps& ops, const std::string& mix) {
    auto nodes = build_body_nodes(ops);
    return emit_body_ops_from_nodes<ENABLE_FUSION>(nodes, mix);
}

inline BodySplit split_body_at_last_write_to(const std::vector<BodyOp>& ops, uint32_t reg_idx) {
    BodySplit s;
    if (ops.empty()) return s;

    const uint32_t target = bit(reg_idx);
    int cut = -1;

    for (int i = 0; i < (int)ops.size(); ++i) {
        if (ops[i].writes_mask & target)
            cut = i;
    }

    if (cut < 0) {
        s.pre = ops;
        return s;
    }

    s.pre.insert(s.pre.end(), ops.begin(), ops.begin() + cut + 1);
    s.post.insert(s.post.end(), ops.begin() + cut + 1, ops.end());
    return s;
}

// Conservative 3-way split for the software-pipelined DAG issue path.
//
// critical: ops before the original MIX[0] cut that must remain before DAG issue
//           either because they produce values needed by MIX[0], or because they
//           do not commute with selected critical ops.
// filler:   pre-cut ops safe to delay until after issuing the next DAG load.
// post:     original post-cut ops, still before delayed MIX[0] commit.
inline BodyTriSplit split_body_for_dag_issue(const std::vector<BodyOp>& ops, uint32_t reg_idx) {
    BodyTriSplit s;
    if (ops.empty()) return s;

    const uint32_t target = bit(reg_idx);
    int cut = -1;

    for (int i = 0; i < (int)ops.size(); ++i) {
        if (ops[i].writes_mask & target)
            cut = i;
    }

    if (cut < 0) {
        s.critical = ops;
        return s;
    }

    std::vector<uint8_t> selected(cut + 1, 0);
    uint32_t need = target;

    bool changed = true;
    while (changed) {
        changed = false;

        // Backward dependency closure: include producers of needed registers.
        for (int i = cut; i >= 0; --i) {
            if ((ops[i].writes_mask & need) && !selected[i]) {
                selected[i] = 1;
                need |= ops[i].reads_mask;
                changed = true;
            }
        }

        // Conservative ordering closure: include earlier non-commuting ops so
        // moving selected critical ops earlier cannot alter program semantics.
        for (int i = 0; i <= cut; ++i) {
            if (selected[i]) continue;

            for (int j = i + 1; j <= cut; ++j) {
                if (selected[j] && ops_conflict(ops[i], ops[j])) {
                    selected[i] = 1;
                    need |= ops[i].reads_mask;
                    changed = true;
                    break;
                }
            }
        }
    }

    for (int i = 0; i <= cut; ++i) {
        if (selected[i]) s.critical.push_back(ops[i]);
        else             s.filler.push_back(ops[i]);
    }

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
    std::string body;
    body += "    uint32_t _c0, _c1, _m, _m1;" + le;
    body += emit_body_ops(build_body_ops(ops, mix), le);
    return body;
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

inline std::string emit_constant_l1_table(const uint32_t* words, size_t n) {
    std::string s;
    s += "__device__ __constant__ uint32_t kawpow_const_l1[PROGPOW_CACHE_WORDS] = {\n";
    for (size_t i = 0; i < n; ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "0x%08xu", words[i]);
        s += "    ";
        s += buf;
        if (i + 1 != n) s += ",";
        s += "\n";
    }
    s += "};\n";
    return s;
}

// ---------------------------------------------------------------------------
// generate_program_macro — emit body macros
// ---------------------------------------------------------------------------
template<bool ENABLE_FUSION = false>
inline std::string generate_program_macro(int block_number) {
    auto ops = collect_ops(block_number);
    auto nodes = build_body_nodes(ops);
    auto body_ops = emit_body_ops_from_nodes<ENABLE_FUSION>(nodes, "MIX");

    auto split = split_body_for_dag_issue(body_ops, 0);

    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "// ProgPoW body macro for period %llu (block %d)\n",
        (unsigned long long)ops.period, block_number);

    std::string c;
    c.reserve(8192);
    c += buf;

    // Full non-pipelined body. This path owns the temp declarations.
    c += "#define PROGPOW_BODY(MIX, DG) do { \\\n";
    c += "    uint32_t _c0, _m; \\\n";
    c += emit_body_ops(body_ops, " \\\n");
    c += emit_dag_merge(ops, "MIX", "DG", " \\\n");
    c += "} while(0)\n\n";

    // 3-split software-pipelined body pieces.
    // These submacros intentionally do NOT declare _c0/_c1/_m/_m1; BODY_PIPE
    // and N-way main loops should declare them once in the caller scope so the
    // compiler can freely reuse the same temporaries.
    c += "#define PROGPOW_BODY_PRECRITICAL(MIX, DG_CUR, IA_RAW_NEXT, MIX0_NEXT, NEXT_LOOP) do { \\\n";
    c += emit_body_ops(split.critical, " \\\n");
    c += emit_dag_prepare_issue_value(ops, "MIX", "DG_CUR", " \\\n");
    c += "    (MIX0_NEXT) = _mix0_next; \\\n";
    c += "    PROGPOW_BPERMUTE_STAGE((MIX0_NEXT), IA_RAW_NEXT, NEXT_LOOP); \\\n";
    c += "} while(0)\n\n";

    c += "#define PROGPOW_BODY_PREFILLER(MIX) do { \\\n";
    c += emit_body_ops(split.filler, " \\\n");
    c += "} while(0)\n\n";

    c += "#define PROGPOW_BODY_POSTISSUE(MIX, DG_CUR, MIX0_NEXT) do { \\\n";
    c += emit_body_ops(split.post, " \\\n");
    c += emit_dag_capture_post("DG_CUR", " \\\n");
    c += "    " + reg("MIX", ops.dag_dsts[0]) + " = (MIX0_NEXT); \\\n";
    c += emit_dag_merge_post(ops, "MIX", " \\\n");
    c += "} while(0)\n\n";

    // Single-hash fallback pipe. N-way loops can call the split pieces directly.
    c += "#define PROGPOW_BODY_PIPE(MIX, DG_CUR, DG_NEXT, NEXT_LOOP) do { \\\n";
    c += "    uint32_t _c0, _m; \\\n";
    c += "    uint32_t _ia_raw_next; \\\n";
    c += "    uint32_t _mix0_next_hold; \\\n";
    c += "    PROGPOW_BODY_PRECRITICAL(MIX, DG_CUR, _ia_raw_next, _mix0_next_hold, NEXT_LOOP); \\\n";
    c += "    PROGPOW_DAG_LOAD_STAGE(_ia_raw_next, DG_NEXT, NEXT_LOOP); \\\n";
    c += "    PROGPOW_BODY_PREFILLER(MIX); \\\n";
    c += "    PROGPOW_BODY_POSTISSUE(MIX, DG_CUR, _mix0_next_hold); \\\n";
    c += "} while(0)\n\n";

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

inline std::string inject_constant_l1_table(const std::string& kernel_source,
                                            const uint32_t* l1_words,
                                            size_t n_words) {
    std::string buf;
    buf.reserve(n_words * 16 + 256);

    buf += "__device__ __constant__ uint32_t kawpow_const_l1[PROGPOW_CACHE_WORDS] = {\n";
    constexpr int PER_LINE = 8;

    for (size_t i = 0; i < n_words; i += PER_LINE) {
        buf += "    ";

        for (int j = 0; j < PER_LINE && (i + j) < n_words; ++j) {
            size_t idx = i + j;

            char tmp[32];
            std::snprintf(tmp, sizeof(tmp), "0x%08xu", l1_words[idx]);
            buf += tmp;

            if (idx + 1 != n_words)
                buf += ", ";
        }

        buf += "\n";
    }
    buf += "};\n";
    buf += "#define KAWPOW_L1_LOAD(IDX) kawpow_const_l1[(IDX)]\n";

    std::string result = kernel_source;
    const std::string marker = "/* KAWPOW_CONST_L1 */";
    auto pos = result.find(marker);
    if (pos != std::string::npos)
        result.replace(pos, marker.size(), buf);
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
template<bool ENABLE_FUSION = false>
inline std::string inject_program(const std::string& kernel_source, int block_number) {
    std::string result = kernel_source;

    const std::string macro_marker = "/* PROGPOW_MACROS */";
    auto pos = result.find(macro_marker);
    if (pos != std::string::npos) {
        result.replace(
            pos,
            macro_marker.size(),
            generate_program_macro<ENABLE_FUSION>(block_number)
        );
    }

    return result;
}

} // namespace kawpow_proggen