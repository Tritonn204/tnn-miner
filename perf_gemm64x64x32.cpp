/*******************************************************************************
 *
 * Standalone i8 GEMM benchmark — 2-Stage Orchestration System
 *
 * Compile:
 *   hipcc -O3 -ffast-math --offload-arch=gfx1100 -std=c++20 -fopenmp -DNDEBUG \
 *     -I/mnt/f/git/rocm-libraries/projects/rocwmma/library/include \
 *     -I/mnt/f/git/Tnn-miner \
 *     -o perf_gemm64x64x32 /mnt/f/git/Tnn-miner/perf_gemm64x64x32.cpp -lrocblas
 *
 * Run:
 *   ./perf_gemm64x64x32 [M [N [K [runs [warmups [alpha [beta]]]]]]]
 *
 ******************************************************************************/

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#ifndef CHECK_ROCBLAS_ERROR
#define CHECK_ROCBLAS_ERROR(expr)                                         \
  do                                                                      \
  {                                                                       \
    if (auto status = (expr); status != rocblas_status_success)           \
    {                                                                     \
      std::cerr << "rocBLAS error: '" << rocblas_status_to_string(status) \
                << "' (" << status << ") at " << __FILE__ << ":"          \
                << __LINE__ << std::endl;                                 \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (false)
#endif

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(expr)                                    \
  do                                                             \
  {                                                              \
    if (auto status = (expr); status != hipSuccess)              \
    {                                                            \
      std::cerr << "HIP error: '" << hipGetErrorString(status)   \
                << "' (" << status << ") at " << __FILE__ << ":" \
                << __LINE__ << std::endl;                        \
      std::exit(EXIT_FAILURE);                                   \
    }                                                            \
  } while (false)
#endif

// ============================================================================
// 2-stage orchestration system include
// ============================================================================

#include "perf_i8gemm_combined.hpp"

// ============================================================================
// Problem descriptor and buffers
// ============================================================================

using InputT  = int8_t;
using OutputT = int32_t;
using ComputeT = int32_t;

struct Problem
{
    uint32_t m = 7168;
    uint32_t n = 7168;
    uint32_t k = 7168;
    ComputeT alpha = 2;
    ComputeT beta = 2;
    uint32_t warmups = 2;
    uint32_t runs = 5;
};

struct Buffers
{
    std::vector<InputT>  hA;
    std::vector<InputT>  hB;
    std::vector<OutputT> hC;
    InputT*  dA = nullptr;
    InputT*  dB = nullptr;
    OutputT* dC = nullptr;
    OutputT* dD = nullptr;
    OutputT* dRef = nullptr;
    OutputT* dRoc = nullptr;
    uint32_t* dMismatch = nullptr;
    uint32_t* dMismatchCount = nullptr;
    uint32_t* dMismatchList = nullptr;
    CombinedOperandProbe* dCombinedProbe = nullptr;
};

enum class TestPattern
{
    Default,
    NegOnePosOne,
    NegOneNegOne,
    Checker,
    Col8Sign
};

inline TestPattern get_test_pattern()
{
    const char* env = std::getenv("TNN_TEST_PATTERN");
    if(!env || !*env)
        return TestPattern::Default;
    const std::string pattern(env);
    if(pattern == "neg1_pos1")
        return TestPattern::NegOnePosOne;
    if(pattern == "neg1_neg1")
        return TestPattern::NegOneNegOne;
    if(pattern == "checker")
        return TestPattern::Checker;
    if(pattern == "col8_sign")
        return TestPattern::Col8Sign;
    return TestPattern::Default;
}

inline const char* test_pattern_name(TestPattern pattern)
{
    switch(pattern)
    {
    case TestPattern::NegOnePosOne: return "neg1_pos1";
    case TestPattern::NegOneNegOne: return "neg1_neg1";
    case TestPattern::Checker: return "checker";
    case TestPattern::Col8Sign: return "col8_sign";
    default: return "default";
    }
}

inline bool sync_each_launch_enabled()
{
    const char* env = std::getenv("TNN_SYNC_EACH_LAUNCH");
    return env && *env && std::string(env) != "0";
}

constexpr uint32_t kMaxMismatchDump = 256u;

inline double gops(const Problem& p)
{
    return 2.0 * static_cast<double>(p.m) * static_cast<double>(p.n) * static_cast<double>(p.k) * 1.0e-9;
}

inline double macs(const Problem& p)
{
    return static_cast<double>(p.m) * static_cast<double>(p.n) * static_cast<double>(p.k) * 1.0e-12;
}

// ============================================================================
// Comparison kernels
// ============================================================================

__global__ void compare_output_kernel(OutputT const* actual,
                                      OutputT const* expected,
                                      uint32_t total,
                                      uint32_t* mismatchIndex)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;
    for(uint32_t i = tid; i < total; i += stride)
    {
        if(actual[i] != expected[i])
            atomicMin(mismatchIndex, i);
    }
}

__global__ void collect_mismatches_kernel(OutputT const* actual,
                                          OutputT const* expected,
                                          uint32_t total,
                                          uint32_t n,
                                          uint32_t* mismatchCount,
                                          uint32_t* mismatchList,
                                          uint32_t maxDump,
                                          uint32_t tileM,
                                          uint32_t tileN)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;
    for(uint32_t i = tid; i < total; i += stride)
    {
        if(actual[i] != expected[i])
        {
            const uint32_t row = i / n;
            const uint32_t col = i % n;
            if(row >= tileM || col >= tileN) continue;
            uint32_t slot = atomicAdd(mismatchCount, 1u);
            if(slot < maxDump) mismatchList[slot] = i;
        }
    }
}

// ============================================================================
// Diff diagnostics
// ============================================================================

void dump_tile_diff(const Problem& p,
                    OutputT const* dActual,
                    OutputT const* dExpected,
                    uint32_t rows = 32,
                    uint32_t cols = 64)
{
    std::vector<OutputT> hActual(static_cast<size_t>(p.m) * p.n);
    std::vector<OutputT> hExpected(static_cast<size_t>(p.m) * p.n);
    CHECK_HIP_ERROR(hipMemcpy(hActual.data(), dActual,
                              hActual.size() * sizeof(OutputT), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hExpected.data(), dExpected,
                              hExpected.size() * sizeof(OutputT), hipMemcpyDeviceToHost));

    const uint32_t viewRows = std::min(rows, p.m);
    const uint32_t viewCols = std::min(cols, p.n);

    std::cout << "--- diff mask first " << viewRows << "x" << viewCols
              << " (. match, X mismatch) ---\n";
    for(uint32_t r = 0; r < viewRows; ++r)
    {
        std::cout << "r" << std::setw(2) << r << " ";
        for(uint32_t c = 0; c < viewCols; ++c)
        {
            uint32_t idx = r * p.n + c;
            std::cout << (hActual[idx] == hExpected[idx] ? '.' : 'X');
            if((c & 15u) == 15u) std::cout << ' ';
        }
        std::cout << '\n';
    }
}

void dump_tile_values(const Problem& p,
                      OutputT const* dActual,
                      OutputT const* dExpected,
                      uint32_t rows = 16,
                      uint32_t cols = 32)
{
    std::vector<OutputT> hActual(static_cast<size_t>(p.m) * p.n);
    std::vector<OutputT> hExpected(static_cast<size_t>(p.m) * p.n);
    CHECK_HIP_ERROR(hipMemcpy(hActual.data(), dActual,
                              hActual.size() * sizeof(OutputT), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hExpected.data(), dExpected,
                              hExpected.size() * sizeof(OutputT), hipMemcpyDeviceToHost));

    std::cout << "--- actual first " << rows << " rows x " << cols << " cols ---\n";
    for(uint32_t r = 0; r < std::min(rows, p.m); ++r)
    {
        std::cout << "r" << r << "  ";
        for(uint32_t c = 0; c < std::min(cols, p.n); ++c)
        {
            std::cout << std::setw(8) << hActual[r * p.n + c] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "--- expected first " << rows << " rows x " << cols << " cols ---\n";
    for(uint32_t r = 0; r < std::min(rows, p.m); ++r)
    {
        std::cout << "r" << r << "  ";
        for(uint32_t c = 0; c < std::min(cols, p.n); ++c)
        {
            std::cout << std::setw(8) << hExpected[r * p.n + c] << ' ';
        }
        std::cout << '\n';
    }
}

void dump_value_origin_map(const Problem& p,
                           OutputT const* dActual,
                           OutputT const* dExpected,
                           uint32_t rows = 16,
                           uint32_t cols = 32)
{
    std::vector<OutputT> hActual(static_cast<size_t>(p.m) * p.n);
    std::vector<OutputT> hExpected(static_cast<size_t>(p.m) * p.n);
    CHECK_HIP_ERROR(hipMemcpy(hActual.data(), dActual,
                              hActual.size() * sizeof(OutputT), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hExpected.data(), dExpected,
                              hExpected.size() * sizeof(OutputT), hipMemcpyDeviceToHost));

    std::cout << "--- origin map actual(r,c) -> expected(r,c), first "
              << rows << "x" << cols << " searched in expected "
              << p.m << "x" << p.n << " ---\n";
    for(uint32_t r = 0; r < std::min(rows, p.m); ++r)
    {
        std::cout << "r" << r << "  ";
        for(uint32_t c = 0; c < std::min(cols, p.n); ++c)
        {
            const OutputT value = hActual[r * p.n + c];
            int foundRow = -1;
            int foundCol = -1;
            bool multi = false;
            for(uint32_t rr = 0; rr < p.m && !multi; ++rr)
            {
                for(uint32_t cc = 0; cc < p.n; ++cc)
                {
                    if(hExpected[rr * p.n + cc] == value)
                    {
                        if(foundRow >= 0)
                        {
                            multi = true;
                            break;
                        }
                        foundRow = static_cast<int>(rr);
                        foundCol = static_cast<int>(cc);
                    }
                }
            }

            if(multi)
            {
                std::cout << "(multi) ";
            }
            else if(foundRow >= 0)
            {
                std::cout << "(" << foundRow << "," << foundCol << ") ";
            }
            else
            {
                std::cout << "(--,--) ";
            }
        }
        std::cout << '\n';
    }
}

void dump_combined_probe(Buffers& buffers)
{
    CombinedOperandProbe probe{};
    CHECK_HIP_ERROR(hipMemcpy(&probe, buffers.dCombinedProbe, sizeof(probe), hipMemcpyDeviceToHost));
    std::cout << "--- combined operand probe ---\n";
    std::cout << "seen=" << probe.seen
              << " warp_m=" << probe.warp_m
              << " warp_n=" << probe.warp_n
              << " lane=" << probe.lane_id
              << " k_substep=" << probe.k_substep
              << " seen_lane8=" << probe.seen_lane8
              << " seen_lane16=" << probe.seen_lane16
              << " seen_lane24=" << probe.seen_lane24 << '\n';
    if(!probe.seen)
        return;
    auto print_arr = [](const char* name, const int32_t* data, uint32_t n) {
        std::cout << name << "=[";
        for(uint32_t i = 0; i < n; ++i)
        {
            if(i) std::cout << ',';
            std::cout << data[i];
        }
        std::cout << "]\n";
    };
    print_arr("a_lo", probe.a_lo, 16);
    print_arr("a_hi", probe.a_hi, 16);
    print_arr("a0", probe.a0, 4);
    print_arr("a1", probe.a1, 4);
    print_arr("a2", probe.a2, 4);
    print_arr("a3", probe.a3, 4);
    print_arr("b0", probe.b0, 4);
    print_arr("b1", probe.b1, 4);
    if(probe.seen_lane8)
    {
        print_arr("a0_lane8", probe.a0_lane8, 4);
        print_arr("a1_lane8", probe.a1_lane8, 4);
        print_arr("a2_lane8", probe.a2_lane8, 4);
        print_arr("a3_lane8", probe.a3_lane8, 4);
        print_arr("b0_lane8", probe.b0_lane8, 4);
    }
    if(probe.seen_lane16)
    {
        print_arr("a0_lane16", probe.a0_lane16, 4);
        print_arr("a1_lane16", probe.a1_lane16, 4);
        print_arr("a2_lane16", probe.a2_lane16, 4);
        print_arr("a3_lane16", probe.a3_lane16, 4);
        print_arr("b0_lane16", probe.b0_lane16, 4);
    }
    if(probe.seen_lane24)
    {
        print_arr("a0_lane24", probe.a0_lane24, 4);
        print_arr("a1_lane24", probe.a1_lane24, 4);
        print_arr("a2_lane24", probe.a2_lane24, 4);
        print_arr("a3_lane24", probe.a3_lane24, 4);
        print_arr("b0_lane24", probe.b0_lane24, 4);
    }
    if(probe.seen_lane15)
    {
        print_arr("a0_lane15", probe.a0_lane15, 4);
        print_arr("a1_lane15", probe.a1_lane15, 4);
        print_arr("a2_lane15", probe.a2_lane15, 4);
        print_arr("a3_lane15", probe.a3_lane15, 4);
        print_arr("b0_lane15", probe.b0_lane15, 4);
    }
    if(probe.seen_lane31)
    {
        print_arr("a0_lane31", probe.a0_lane31, 4);
        print_arr("a1_lane31", probe.a1_lane31, 4);
        print_arr("a2_lane31", probe.a2_lane31, 4);
        print_arr("a3_lane31", probe.a3_lane31, 4);
        print_arr("b0_lane31", probe.b0_lane31, 4);
    }
    if(probe.seen_acc_lane0) print_arr("acc_lane0", probe.acc_lane0, 8);
    if(probe.seen_acc_lane8) print_arr("acc_lane8", probe.acc_lane8, 8);
    if(probe.seen_acc_lane16) print_arr("acc_lane16", probe.acc_lane16, 8);
    if(probe.seen_acc_lane24) print_arr("acc_lane24", probe.acc_lane24, 8);
    if(probe.seen_lane15) print_arr("acc_lane15", probe.acc_lane15, 8);
    if(probe.seen_lane31) print_arr("acc_lane31", probe.acc_lane31, 8);
    if(probe.seen_k1_lane0) print_arr("b0_k1_lane0", probe.b0_k1_lane0, 4);
    if(probe.seen_k1_lane8) print_arr("b0_k1_lane8", probe.b0_k1_lane8, 4);
    if(probe.seen_k1_lane16) print_arr("b0_k1_lane16", probe.b0_k1_lane16, 4);
    if(probe.seen_k1_lane24) print_arr("b0_k1_lane24", probe.b0_k1_lane24, 4);
    print_arr("dbg_a0b0_lane0", probe.dbg_a0b0_lane0, 8);
    print_arr("dbg_a1b0_lane0", probe.dbg_a1b0_lane0, 8);
    print_arr("dbg_a2b0_lane0", probe.dbg_a2b0_lane0, 8);
    print_arr("dbg_a3b0_lane0", probe.dbg_a3b0_lane0, 8);
    print_arr("dbg_a0b0_lane16", probe.dbg_a0b0_lane16, 8);
    print_arr("dbg_a1b0_lane16", probe.dbg_a1b0_lane16, 8);
    print_arr("dbg_a2b0_lane16", probe.dbg_a2b0_lane16, 8);
    print_arr("dbg_a3b0_lane16", probe.dbg_a3b0_lane16, 8);
    if(probe.seen_lane15)
    {
        print_arr("dbg_a0b0_lane15", probe.dbg_a0b0_lane15, 8);
        print_arr("dbg_a1b0_lane15", probe.dbg_a1b0_lane15, 8);
        print_arr("dbg_a2b0_lane15", probe.dbg_a2b0_lane15, 8);
        print_arr("dbg_a3b0_lane15", probe.dbg_a3b0_lane15, 8);
    }
    if(probe.seen_lane31)
    {
        print_arr("dbg_a0b0_lane31", probe.dbg_a0b0_lane31, 8);
        print_arr("dbg_a1b0_lane31", probe.dbg_a1b0_lane31, 8);
        print_arr("dbg_a2b0_lane31", probe.dbg_a2b0_lane31, 8);
        print_arr("dbg_a3b0_lane31", probe.dbg_a3b0_lane31, 8);
    }
}

void dump_gfx11_store_coverage_16x16()
{
    constexpr uint32_t M = 16u;
    constexpr uint32_t N = 16u;
    int counts[M][N] = {};

    std::cout << "--- gfx11 16x16 store coverage ---\n";
    for(uint32_t lane_id = 0; lane_id < 32u; ++lane_id)
    {
        const uint32_t row_in_blk = lane_id & 0xFu;
        const uint32_t lane_half  = lane_id >> 4;
        for(uint32_t e = 0; e < 8u; ++e)
        {
            const uint32_t out_row = lane_half * 8u + (e & 0x3u) * 2u + (row_in_blk & 0x1u);
            const uint32_t out_col = (row_in_blk >> 2) * 4u + ((row_in_blk >> 1) & 0x1u) * 2u + (e >> 2);
            if(out_row < M && out_col < N)
                counts[out_row][out_col] += 1;
        }
    }

    for(uint32_t r = 0; r < M; ++r)
    {
        std::cout << "r" << std::setw(2) << r << " ";
        for(uint32_t c = 0; c < N; ++c)
            std::cout << counts[r][c];
        std::cout << '\n';
    }
}

void dump_gfx11_lane_decode_15_31()
{
    std::cout << "--- gfx11 lane decode 15/31 ---\n";
    for(uint32_t lane_id : {15u, 31u})
    {
        const uint32_t row_in_blk = lane_id & 0xFu;
        const uint32_t lane_half  = lane_id >> 4;
        std::cout << "lane " << lane_id << ": ";
        for(uint32_t e = 0; e < 8u; ++e)
        {
            const uint32_t out_row = lane_half * 8u + (e & 0x3u) * 2u + (row_in_blk & 0x1u);
            const uint32_t out_col = (row_in_blk >> 2) * 4u + ((row_in_blk >> 1) & 0x1u) * 2u + (e >> 2);
            std::cout << "e" << e << "->(" << out_row << "," << out_col << ") ";
        }
        std::cout << '\n';
    }
}

void dump_probe_acc_origins(const Problem& p, Buffers& buffers, OutputT const* dExpected)
{
    CombinedOperandProbe probe{};
    CHECK_HIP_ERROR(hipMemcpy(&probe, buffers.dCombinedProbe, sizeof(probe), hipMemcpyDeviceToHost));
    std::vector<OutputT> hExpected(static_cast<size_t>(p.m) * p.n);
    CHECK_HIP_ERROR(hipMemcpy(hExpected.data(), dExpected,
                              hExpected.size() * sizeof(OutputT), hipMemcpyDeviceToHost));

    struct LaneView
    {
        const char* name;
        const int32_t* vals;
    };

    const LaneView views[] = {
        {"acc_lane0", probe.acc_lane0},
        {"acc_lane8", probe.acc_lane8},
        {"acc_lane15", probe.acc_lane15},
        {"acc_lane16", probe.acc_lane16},
        {"acc_lane24", probe.acc_lane24},
        {"acc_lane31", probe.acc_lane31},
    };

    std::cout << "--- probe acc origins ---\n";
    for(const auto& view : views)
    {
        std::cout << view.name << ": ";
        for(uint32_t e = 0; e < 8u; ++e)
        {
            const OutputT value = view.vals[e];
            int foundRow = -1;
            int foundCol = -1;
            bool multi = false;
            for(uint32_t r = 0; r < p.m && !multi; ++r)
            {
                for(uint32_t c = 0; c < p.n; ++c)
                {
                    if(hExpected[r * p.n + c] == value)
                    {
                        if(foundRow >= 0)
                        {
                            multi = true;
                            break;
                        }
                        foundRow = static_cast<int>(r);
                        foundCol = static_cast<int>(c);
                    }
                }
            }

            if(multi)
                std::cout << "e" << e << "=(multi) ";
            else if(foundRow >= 0)
                std::cout << "e" << e << "=(" << foundRow << "," << foundCol << ") ";
            else
                std::cout << "e" << e << "=(--,--) ";
        }
        std::cout << '\n';
    }
}

// ============================================================================
// compare_output — returns true if actual matches expected.
// ============================================================================

bool compare_output(const Problem& p, OutputT const* dActual, OutputT const* dExpected, Buffers& buffers)
{
    const uint32_t total = p.m * p.n;
    const uint32_t noMismatch = 0xffffffffu;
    CHECK_HIP_ERROR(hipMemcpy(buffers.dMismatch, &noMismatch, sizeof(uint32_t), hipMemcpyHostToDevice));
    compare_output_kernel<<<1024, 256>>>(dActual, dExpected, total, buffers.dMismatch);
    CHECK_HIP_ERROR(hipGetLastError());
    uint32_t mismatch = noMismatch;
    CHECK_HIP_ERROR(hipMemcpy(&mismatch, buffers.dMismatch, sizeof(uint32_t), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    if(mismatch != noMismatch)
    {
        OutputT actual = 0, expected = 0;
        CHECK_HIP_ERROR(hipMemcpy(&actual, dActual + mismatch, sizeof(OutputT), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(&expected, dExpected + mismatch, sizeof(OutputT), hipMemcpyDeviceToHost));
        std::cout << " first mismatch idx=" << mismatch
                  << " row=" << (mismatch / p.n)
                  << " col=" << (mismatch % p.n)
                  << " actual=" << actual
                  << " expected=" << expected << std::endl;

        uint32_t zero = 0;
        CHECK_HIP_ERROR(hipMemcpy(buffers.dMismatchCount, &zero, sizeof(uint32_t), hipMemcpyHostToDevice));
        collect_mismatches_kernel<<<1024, 256>>>(dActual, dExpected, total, p.n,
                                                  buffers.dMismatchCount, buffers.dMismatchList,
                                                  kMaxMismatchDump, 64u, 64u);
        CHECK_HIP_ERROR(hipGetLastError());
        CHECK_HIP_ERROR(hipDeviceSynchronize());
    }
    return mismatch == noMismatch;
}

inline bool compare_output(const Problem& p, Buffers& buffers)
{
    return compare_output(p, buffers.dD, buffers.dRef, buffers);
}

// ============================================================================
// init_buffers / free_buffers
// ============================================================================

void init_buffers(const Problem& p, Buffers& buffers)
{
    buffers.hA.resize(static_cast<size_t>(p.m) * p.k);
    buffers.hB.resize(static_cast<size_t>(p.k) * p.n);
    buffers.hC.resize(static_cast<size_t>(p.m) * p.n);
    const TestPattern pattern = get_test_pattern();

    auto make_i8 = [](uint32_t x) -> InputT {
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        return static_cast<InputT>(static_cast<int32_t>(x & 0xffU) - 128);
    };

    for(uint32_t kk = 0; kk < p.k; ++kk)
    {
        for(uint32_t row = 0; row < p.m; ++row)
        {
            size_t i = row + kk * static_cast<size_t>(p.m);
            switch(pattern)
            {
            case TestPattern::NegOnePosOne:
            case TestPattern::NegOneNegOne:
            case TestPattern::Col8Sign:
                buffers.hA[i] = static_cast<InputT>(-1);
                break;
            case TestPattern::Checker:
                buffers.hA[i] = ((row + kk) & 1u) ? static_cast<InputT>(-1) : static_cast<InputT>(1);
                break;
            default:
                buffers.hA[i] = make_i8(0xA501u ^ row * 0x1f123bb5u ^ kk * 0x9e3779b9u);
                break;
            }
        }
    }

    for(uint32_t col = 0; col < p.n; ++col)
    {
        for(uint32_t kk = 0; kk < p.k; ++kk)
        {
            size_t i = kk + col * static_cast<size_t>(p.k);
            switch(pattern)
            {
            case TestPattern::NegOnePosOne:
                buffers.hB[i] = static_cast<InputT>(1);
                break;
            case TestPattern::NegOneNegOne:
                buffers.hB[i] = static_cast<InputT>(-1);
                break;
            case TestPattern::Col8Sign:
                buffers.hB[i] = ((col >> 3) & 1u) ? static_cast<InputT>(-1) : static_cast<InputT>(1);
                break;
            case TestPattern::Checker:
                buffers.hB[i] = ((kk + col) & 1u) ? static_cast<InputT>(1) : static_cast<InputT>(-1);
                break;
            default:
                buffers.hB[i] = make_i8(0xB701u ^ kk * 0x85ebca6bu ^ col * 0xc2b2ae35u);
                break;
            }
        }
    }

    for(size_t i = 0; i < buffers.hC.size(); ++i)
    {
        buffers.hC[i] = (pattern == TestPattern::Default)
            ? static_cast<OutputT>((i * 7 + 5) & 0x7f)
            : static_cast<OutputT>(0);
    }

    if(pattern != TestPattern::Default)
    {
        std::cout << "Using test pattern: " << test_pattern_name(pattern)
                  << " (C forced to zero)" << std::endl;
    }

    CHECK_HIP_ERROR(hipMalloc(&buffers.dA, buffers.hA.size() * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&buffers.dB, buffers.hB.size() * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&buffers.dC, buffers.hC.size() * sizeof(OutputT)));
    CHECK_HIP_ERROR(hipMalloc(&buffers.dD, buffers.hC.size() * sizeof(OutputT)));
    CHECK_HIP_ERROR(hipMalloc(&buffers.dRef, buffers.hC.size() * sizeof(OutputT)));
    CHECK_HIP_ERROR(hipMalloc(&buffers.dRoc, buffers.hC.size() * sizeof(OutputT)));
    CHECK_HIP_ERROR(hipMalloc(&buffers.dMismatch, sizeof(uint32_t)));
    CHECK_HIP_ERROR(hipMalloc(&buffers.dMismatchCount, sizeof(uint32_t)));
    CHECK_HIP_ERROR(hipMalloc(&buffers.dMismatchList, kMaxMismatchDump * sizeof(uint32_t)));
    CHECK_HIP_ERROR(hipMalloc(&buffers.dCombinedProbe, sizeof(CombinedOperandProbe)));

    CHECK_HIP_ERROR(hipMemcpy(buffers.dA, buffers.hA.data(), buffers.hA.size() * sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(buffers.dB, buffers.hB.data(), buffers.hB.size() * sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(buffers.dC, buffers.hC.data(), buffers.hC.size() * sizeof(OutputT), hipMemcpyHostToDevice));
}

void free_buffers(Buffers& buffers)
{
    CHECK_HIP_ERROR(hipFree(buffers.dA));
    CHECK_HIP_ERROR(hipFree(buffers.dB));
    CHECK_HIP_ERROR(hipFree(buffers.dC));
    CHECK_HIP_ERROR(hipFree(buffers.dD));
    CHECK_HIP_ERROR(hipFree(buffers.dRef));
    CHECK_HIP_ERROR(hipFree(buffers.dRoc));
    CHECK_HIP_ERROR(hipFree(buffers.dMismatch));
    CHECK_HIP_ERROR(hipFree(buffers.dMismatchCount));
    CHECK_HIP_ERROR(hipFree(buffers.dMismatchList));
    CHECK_HIP_ERROR(hipFree(buffers.dCombinedProbe));
}

std::optional<KernelVariant> get_forced_variant()
{
    const char* env = std::getenv("TNN_ONLY_VARIANT");
    if(!env || !*env)
        return std::nullopt;
    const std::string want(env);
    for(const auto& info : k_VariantTable)
    {
        if(want == variant_name(info.id))
            return info.id;
    }
    std::cerr << "Unknown TNN_ONLY_VARIANT='" << want << "'" << std::endl;
    std::exit(EXIT_FAILURE);
}

// ============================================================================
// rocBLAS reference runner
// ============================================================================

void run_rocblas(const Problem& p, Buffers& buffers)
{
    rocblas_handle handle{};
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));
    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    ComputeT alpha_copy = p.alpha;
    ComputeT beta_copy  = p.beta;
    auto launch = [&]() {
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(handle,
                                            rocblas_operation_none,
                                            rocblas_operation_none,
                                            p.m, p.n, p.k,
                                            &alpha_copy,
                                            buffers.dA, rocblas_datatype_i8_r, p.m,
                                            buffers.dB, rocblas_datatype_i8_r, p.k,
                                            &beta_copy,
                                            buffers.dC, rocblas_datatype_i32_r, p.m,
                                            buffers.dRoc, rocblas_datatype_i32_r, p.m,
                                            rocblas_datatype_i32_r,
                                            rocblas_gemm_algo_standard, 0, 0));
    };

    CHECK_HIP_ERROR(hipMemset(buffers.dRoc, 0, buffers.hC.size() * sizeof(OutputT)));
    for(uint32_t i = 0; i < p.warmups; ++i) { launch(); }
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    hipEvent_t startEvent{}, stopEvent{};
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
    CHECK_HIP_ERROR(hipEventRecord(startEvent));
    for(uint32_t i = 0; i < p.runs; ++i) { launch(); }
    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    float elapsedMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));

    auto stats = compute_stats(p.m, p.n, p.k, p.runs, elapsedMs);
    print_result("rocBLAS_i8", true, 0, 0, 0, 0, 0, stats);
}

// ============================================================================
// Benchmark a single kernel variant against rocBLAS reference
// ============================================================================

bool run_variant(KernelVariant v, const Problem& p, Buffers& buffers, OutputT const* dExpected)
{
    if(!variant_compatible(v, p.m, p.n, p.k))
    {
        std::cout << "  " << variant_name(v) << " skipped: dims not divisible by tile ("
                  << variant_tile_m(v) << "x" << variant_tile_n(v) << "x"
                  << variant_tile_k(v) << ")" << std::endl;
        return false;
    }

    uint32_t tile_m = variant_tile_m(v);
    uint32_t tile_n = variant_tile_n(v);
    uint32_t unroll = variant_unroll_k(v);
    uint32_t lds_kb = variant_lds_bytes(v) / 1024u;
    uint32_t tile_k = variant_tile_k(v);

    int maxSharedBytes = 0;
    CHECK_HIP_ERROR(hipDeviceGetAttribute(&maxSharedBytes, hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
    if(static_cast<int>(variant_lds_bytes(v)) > maxSharedBytes)
    {
        std::cout << "  " << variant_name(v) << " skipped: LDS " << variant_lds_bytes(v)
                  << " > " << static_cast<uint32_t>(maxSharedBytes) << std::endl;
        return false;
    }

    auto launch = [&]() {
        CHECK_HIP_ERROR(hipMemset(buffers.dCombinedProbe, 0, sizeof(CombinedOperandProbe)));
        launch_dispatch(v, p.m, p.n, p.k,
                        buffers.dA, buffers.dB,
                        buffers.dC, buffers.dD,
                        p.m, p.k, p.n, p.n,
                        p.alpha, p.beta, buffers.dCombinedProbe, 0);
        CHECK_HIP_ERROR(hipGetLastError());
        if(sync_each_launch_enabled())
            CHECK_HIP_ERROR(hipDeviceSynchronize());
    };

    CHECK_HIP_ERROR(hipMemset(buffers.dD, 0, buffers.hC.size() * sizeof(OutputT)));
    for(uint32_t i = 0; i < p.warmups; ++i) { launch(); }
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    hipEvent_t startEvent{}, stopEvent{};
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
    CHECK_HIP_ERROR(hipEventRecord(startEvent));
    for(uint32_t i = 0; i < p.runs; ++i) { launch(); }
    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    float elapsedMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    bool ok = compare_output(p, buffers.dD, dExpected, buffers);
    if(!ok)
    {
        dump_tile_diff(p, buffers.dD, dExpected, 32, 64);
        dump_tile_values(p, buffers.dD, dExpected, 16, 32);
        dump_value_origin_map(p, buffers.dD, dExpected, 16, 32);
        dump_combined_probe(buffers);
        dump_gfx11_store_coverage_16x16();
        dump_gfx11_lane_decode_15_31();
        dump_probe_acc_origins(p, buffers, dExpected);
    }
    auto stats = compute_stats(p.m, p.n, p.k, p.runs, elapsedMs);
    print_result(variant_name(v), ok, tile_m, tile_n, tile_k, unroll, lds_kb, stats);
    return true;
}

// ============================================================================
// main — test all compatible variants, show auto-selection
// ============================================================================

int main(int argc, char** argv)
{
    Problem p{};
    if(argc >= 4)
    {
        p.m = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10));
        p.n = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
        p.k = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
    }
    if(argc >= 6)
    {
        p.runs = static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10));
        p.warmups = static_cast<uint32_t>(std::strtoul(argv[5], nullptr, 10));
    }
    if(argc >= 8)
    {
        p.alpha = static_cast<ComputeT>(std::strtol(argv[6], nullptr, 10));
        p.beta  = static_cast<ComputeT>(std::strtol(argv[7], nullptr, 10));
    }

    std::cout << "dims M=" << p.m << " N=" << p.n << " K=" << p.k
              << " runs=" << p.runs << " warmups=" << p.warmups
              << " alpha=" << p.alpha << " beta=" << p.beta << std::endl;

    Buffers buffers{};
    init_buffers(p, buffers);

    std::cout << "\n--- Running rocBLAS reference ---" << std::endl;
    print_combined_header();
    run_rocblas(p, buffers);

    std::cout << "\n--- Testing All Compatible Variants ---" << std::endl;
    print_combined_header();

    uint32_t tested = 0;
    uint32_t passed = 0;
    std::optional<KernelVariant> forced = get_forced_variant();

    for(const auto& info : k_VariantTable)
    {
        if(forced && info.id != *forced)
            continue;
        bool ok = run_variant(info.id, p, buffers, buffers.dRoc);
        if(variant_compatible(info.id, p.m, p.n, p.k))
        {
            ++tested;
            if(ok) ++passed;
        }
    }

    std::cout << "\nResults: " << passed << "/" << tested << " compatible variants passed." << std::endl;

    KernelVariant best = select_best_variant(p.m, p.n, p.k);
    std::cout << "Auto-selected best variant: " << variant_name(best)
              << " (tile " << variant_tile_m(best) << "x" << variant_tile_n(best)
              << "x" << variant_tile_k(best) << " U" << variant_unroll_k(best) << ")" << std::endl;

    free_buffers(buffers);
    return 0;
}
