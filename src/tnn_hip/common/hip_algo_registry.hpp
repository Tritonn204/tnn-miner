#pragma once
#include "gpu_algo_impl.hpp"
#include "aligned_alloc.hip.hpp"
#include "oro_seh_wrappers.hpp"
#include <memory>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "tnn_hip_common_embedded.hpp"
#ifdef TNN_XELISHASH
#include "xelis_embedded_headers.hpp"
#include "xelis-hash-v3.hip.hpp"
#include <tnn_hip/crypto/xelis-hash/newton-lut-config.hip.inc>
#endif

#ifdef TNN_KAWPOW
#include "kawpow.hip.hpp"
#include "kawpow_embedded_headers.hpp"
#include "ethash-dag-gen.hip.hpp"
#include <algo_definitions.h>
#include <ethash/ethash.hpp>
#include <ethash/ethash-internal.hpp>
#include <ethash/kawpow_coins.h>
#include <crypto/kawpow/kawpow_algo.h>
#include "../crypto/kawpow/kawpow_proggen.hpp"
#endif

// ============================================================================
// Xelis V3 Shared Memory Calculator
// ============================================================================
inline size_t xelis_v3_shared_mem(int block_size)
{
  return 0; // AES now uses __ldg (NVIDIA) or __constant__ col1 (AMD), no LDS sbox
}

// ============================================================================
// Wavefront/warp size detection — used for stage1 and blake3 cooperative TPB
// ============================================================================
static inline int xelis_get_warp_size(int device_id)
{
  oroDeviceProp_t props{};
  if (oroGetDeviceProperties(&props, tnn_get_device(device_id)) == oroSuccess)
    return props.warpSize;
  return 32; // safe default
}

// Matches compile-time XELIS_STAGE1_COOP_TPB: wave32→32, wave64→64
static inline int xelis_stage1_coop_tpb(int device_id)
{
  return xelis_get_warp_size(device_id);
}

// Matches compile-time XELIS_B3_COOP_TPB: wave32→96, wave64→64
static inline int xelis_b3_coop_tpb(int device_id)
{
  return xelis_get_warp_size(device_id) == 32 ? 96 : 64;
}

// ============================================================================
// Xelis V3 Strategy Definitions
// ============================================================================

// Strategy indices (opaque to the generic tune system)
enum class XelisStrategy : uint8_t
{
  Mono = 0,     // s1+s3_hybrid_v2+b3 monolithic
  Baseline = 1, // s1+s3_hybrid_v2 fused, b3 separate
  Sep = 2,      // all 3 separate (warp-coop b3)
  Neo = 3,      // s1 separate, s3+b3 fused
  SepP = 4,     // all 3 separate (smem pipelined b3 @ 512 TPB)
};

// ============================================================================
// Xelis V3 4-Strategy Execution Dispatch
// ============================================================================

// Helper: choose which stage1 kernel to use based on GPU capabilities
static inline const char *xelis_pick_stage1(int dev)
{
  // bool cooperative = false;
  // if (tnn_is_nvidia_device(dev)) {
  //     cooperative = is_nvidia_ampere_plus(dev);
  // } else {
  //     if (is_amd_rdna_plus(dev)) {
  //         cooperative = true;
  //     } else {
  //         oroDeviceProp_t props{};
  //         if (oroGetDeviceProperties(&props, tnn_get_device(dev)) == oroSuccess) {
  //             const int gfx = parse_gfx_number(props.gcnArchName);
  //             // Vega (900), RVII (906), MI200 (90, from gfx90a), MI300 (940-942)
  //             cooperative = (gfx == 900 || gfx == 906 || gfx == 90 ||
  //                            (gfx >= 940 && gfx <= 942));
  //         }
  //     }
  // }
  // return cooperative ? "xelis_stage1_cooperative" : "xelis_stage1_kernel";
  return "xelis_stage1_cooperative";
}

// Helper: launch a single stage1 chunk (internal).
// scratch_offset = global hash index of this chunk's first hash.
// The kernel uses scratch_offset to compute global_idx for both
// scratch memory indexing and nonce derivation.
static inline bool xelis_launch_stage1_chunk(
    oroFunction_t kernel,
    const char *stage1_name,
    const KernelLaunchContext &ctx,
    uint32_t chunk_batch,
    uint32_t scratch_offset,
    int stage1_block_size,
    size_t shared_mem)
{
  int num_blocks = (chunk_batch + stage1_block_size - 1) / stage1_block_size;  // round up; serial kernel has batch_size guard

  void *args[] = {
      (void *)&ctx.d_input,
      (void *)&ctx.d_scratch,
      (void *)&ctx.nonce_start,
      (void *)&chunk_batch,
      (void *)&scratch_offset};

  TNN_LOG_TRACE("[LAUNCH] Stage1(%s): kernel=%p, grid=%d, block=%d, shared=%zu, offset=%u, batch=%u\n",
                stage1_name, (void *)kernel, num_blocks, stage1_block_size, shared_mem,
                scratch_offset, chunk_batch);
  fflush(stdout);

  oroError_t err = oro_safe_launch(
      kernel,
      num_blocks, 1, 1,
      stage1_block_size, 1, 1,
      shared_mem, ctx.stream,
      args, nullptr);

  if (err != oroSuccess)
  {
    TNN_LOG_ERROR("[XELIS] Stage1 (%s) launch failed: %s\n",
                  stage1_name, tnn_error_string(err));
    return false;
  }
  return true;
}

// Helper: launch stage1 kernel (split-batch if s1_knee_batch tune key is set)
static inline bool xelis_launch_stage1(
    const KernelMap &kernels,
    const KernelLaunchContext &ctx,
    int dev)
{
  const char *stage1_name = xelis_pick_stage1(dev);
  auto it = kernels.find(stage1_name);
  if (it == kernels.end())
    return false;

  bool cooperative = (std::strcmp(stage1_name, "xelis_stage1_cooperative") == 0);

  int s1_tpb = (int)ctx.get_tune_key("s1_tpb", cooperative ? xelis_stage1_coop_tpb(dev) : 32);
  int stage1_block_size = s1_tpb;
  size_t shared_mem = cooperative ? ((size_t)s1_tpb * 176) : 0;

#ifdef WITH_OROCHI
  {
    extern thipModuleLaunchKernel *hipModuleLaunchKernel;
    TNN_LOG_TRACE("[LAUNCH] hipModuleLaunchKernel fptr = %p\n", (void *)hipModuleLaunchKernel);
  }
#endif

  // Check for s1 bandwidth knee — split into chunks to avoid BW cliff
  uint32_t knee = (uint32_t)ctx.get_tune_key("s1_knee_batch", 0);
  // Align knee down to block size
  if (knee > 0)
    knee = (knee / stage1_block_size) * stage1_block_size;

  if (knee > 0 && knee < ctx.batch_size)
  {
    // Use scratch_offset to tell each chunk where it lives in the global
    // scratch array. Same base pointer, same nonce_start — the kernel
    // computes global_idx = local_tid + scratch_offset, which indexes
    // both the scratchpad (× XELIS_MEMORY_SIZE_V3) and the nonce.
    uint32_t done = 0;

    while (done < ctx.batch_size)
    {
      uint32_t chunk = std::min(knee, ctx.batch_size - done);

      if (!xelis_launch_stage1_chunk(
              it->second, stage1_name, ctx,
              chunk, /*scratch_offset=*/done,
              stage1_block_size, shared_mem))
        return false;

      done += chunk;
    }

    TNN_LOG_TRACE("[LAUNCH] Stage1 split-batch: %u hashes in %u-hash chunks\n",
                  ctx.batch_size, knee);
    fflush(stdout);
    return true;
  }

  // Single launch (no knee or batch <= knee)
  return xelis_launch_stage1_chunk(
      it->second, stage1_name, ctx,
      ctx.batch_size, /*scratch_offset=*/0,
      stage1_block_size, shared_mem);
}

// Helper: launch blake3 batch kernel
static inline bool xelis_launch_blake3(
    const KernelMap &kernels,
    const KernelLaunchContext &ctx,
    int dev)
{
  uint32_t scratch_offset = 0;

  // Try cooperative blake3 first (global-CV: no smem, higher occupancy)
  // Launch: <<<batch_size, XELIS_B3_COOP_TPB>>> — 1 block per hash
  // CV workspace lives at end of scratch allocation (scratch_per_hash includes 8512B)
  auto it_wc = kernels.find("xelis_blake3_warp_coop_batch");
  if (it_wc != kernels.end())
  {
    // Use tuned b3_tpb if available, else fall back to arch default
    int blake3_block_size = (int)ctx.get_tune_key("b3_tpb", xelis_b3_coop_tpb(dev));
    int blake3_num_blocks = ctx.batch_size; // 1 block per hash

    void *args[] = {
        (void *)&ctx.d_scratch,
        (void *)&ctx.d_outputs,
        (void *)&ctx.batch_size,
        (void *)&scratch_offset,
        (void *)&ctx.d_difficulty_target,
        (void *)&ctx.d_solutions,
        (void *)&ctx.nonce_start};

    TNN_LOG_TRACE("[LAUNCH] Blake3 warp-coop: kernel=%p, grid=%d, block=%d\n",
                  (void *)it_wc->second, blake3_num_blocks, blake3_block_size);
    fflush(stdout);

    oroError_t err = oro_safe_launch(
        it_wc->second,
        blake3_num_blocks, 1, 1,
        blake3_block_size, 1, 1,
        0, ctx.stream,
        args, nullptr);

    if (err == oroSuccess)
      return true;
    TNN_LOG_ERROR("[XELIS] blake3_warp_coop launch failed (%s), falling back to opt\n",
                  tnn_error_string(err));
  }

  // Fallback 1: per-thread optimized blake3 (shared cv_stack)
  auto it_opt = kernels.find("xelis_blake3_opt_batch");
  if (it_opt != kernels.end())
  {
    int blake3_block_size = 32;
    size_t smem = (size_t)blake3_block_size * 10 * 8 * sizeof(uint32_t);
    int blake3_num_blocks = (ctx.batch_size + blake3_block_size - 1) / blake3_block_size;

    void *args[] = {
        (void *)&ctx.d_scratch,
        (void *)&ctx.d_outputs,
        (void *)&ctx.batch_size,
        (void *)&scratch_offset,
        (void *)&ctx.d_difficulty_target,
        (void *)&ctx.d_solutions,
        (void *)&ctx.nonce_start};

    TNN_LOG_TRACE("[LAUNCH] Blake3 OPT: kernel=%p, grid=%d, block=%d, smem=%zu\n",
                  (void *)it_opt->second, blake3_num_blocks, blake3_block_size, smem);
    fflush(stdout);

    oroError_t err = oro_safe_launch(
        it_opt->second,
        blake3_num_blocks, 1, 1,
        blake3_block_size, 1, 1,
        smem, ctx.stream,
        args, nullptr);

    if (err == oroSuccess)
      return true;
    TNN_LOG_ERROR("[XELIS] blake3_opt launch failed (%s), falling back to original\n",
                  tnn_error_string(err));
  }

  // Fallback: original blake3 batch kernel
  auto it = kernels.find("xelis_blake3_batch");
  if (it == kernels.end())
    return false;

  int blake3_block_size = 256;
  int blake3_num_blocks = (ctx.batch_size + blake3_block_size - 1) / blake3_block_size;

  void *args[] = {
      (void *)&ctx.d_scratch,
      (void *)&ctx.d_outputs,
      (void *)&ctx.batch_size,
      (void *)&scratch_offset,
      (void *)&ctx.d_difficulty_target,
      (void *)&ctx.d_solutions,
      (void *)&ctx.nonce_start};

  TNN_LOG_TRACE("[LAUNCH] Blake3 fallback: kernel=%p, grid=%d, block=%d\n",
                (void *)it->second, blake3_num_blocks, blake3_block_size);
  fflush(stdout);

  oroError_t err = oro_safe_launch(
      it->second,
      blake3_num_blocks, 1, 1,
      blake3_block_size, 1, 1,
      0, ctx.stream,
      args, nullptr);

  TNN_LOG_TRACE("[LAUNCH] Blake3 returned %d (%s)\n", (int)err, tnn_error_string(err));
  fflush(stdout);

  if (err != oroSuccess)
  {
    TNN_LOG_ERROR("[XELIS] Blake3 launch failed: %s\n", tnn_error_string(err));
    return false;
  }
  return true;
}

// Helper: launch smem-based pipelined blake3 (tuned TPB, 1 block per hash)
// Falls back to xelis_launch_blake3() if kernel not found or launch fails.
static inline bool xelis_launch_blake3_smem(
    const KernelMap &kernels,
    const KernelLaunchContext &ctx,
    int dev)
{
  auto it = kernels.find("xelis_blake3_smem_batch");
  if (it != kernels.end())
  {
    int tpb = (int)ctx.get_tune_key("b3_tpb", 512);
    uint32_t scratch_offset = 0;
    void *args[] = {
        (void *)&ctx.d_scratch,
        (void *)&ctx.d_outputs,
        (void *)&ctx.batch_size,
        (void *)&scratch_offset,
        (void *)&ctx.d_difficulty_target,
        (void *)&ctx.d_solutions,
        (void *)&ctx.nonce_start};

    TNN_LOG_TRACE("[LAUNCH] Blake3 smem: kernel=%p, grid=%d, block=%d\n",
                  (void *)it->second, ctx.batch_size, tpb);
    fflush(stdout);

    oroError_t err = oro_safe_launch(
        it->second,
        ctx.batch_size, 1, 1,
        tpb, 1, 1,
        0, ctx.stream,
        args, nullptr);

    if (err == oroSuccess)
      return true;
    TNN_LOG_ERROR("[XELIS] blake3_smem launch failed (%s), falling back\n",
                  tnn_error_string(err));
  }

  return xelis_launch_blake3(kernels, ctx, dev);
}

// ============================================================================
// Xelis V3 Tune Key Probe — s1 bandwidth knee detection
// ============================================================================
inline bool xelis_tune_key_probe(
    const KernelMap &kernels,
    const oroDeviceProp_t &device_props,
    int compute_units,
    oroStream_t stream,
    const AlgoConfig &config,
    TuningResult &result,
    int device_id)
{
  // Sep/SepP use separate s1 + separate blake3
  // Neo uses separate s1 only (blake3 fused in s3b3)
  // Mono/Baseline don't use separate s1
  auto strat = static_cast<XelisStrategy>(result.strategy);
  bool has_separate_s1 = (strat == XelisStrategy::Sep || strat == XelisStrategy::Neo || strat == XelisStrategy::SepP);
  bool has_separate_b3 = (strat == XelisStrategy::Sep || strat == XelisStrategy::SepP || strat == XelisStrategy::Baseline);
  if (!has_separate_s1 && !has_separate_b3)
    return true; // no probe needed

  if (!has_separate_s1)
    goto blake3_sweep;

  {
  const char *s1_name = xelis_pick_stage1(device_id);
  auto it = kernels.find(s1_name);
  if (it == kernels.end())
    return false;

  bool cooperative = (std::strcmp(s1_name, "xelis_stage1_cooperative") == 0);

  // ── S1 TPB sweep ──
  // Find optimal TPB at the winning batch size, then use it for the knee sweep.
  int best_s1_tpb = cooperative ? xelis_stage1_coop_tpb(device_id) : 32;
  {
    uint8_t *tpb_input = nullptr;
    (void)oro_safe_malloc((oroDeviceptr *)&tpb_input, config.template_size);
    if (!tpb_input)
      return false;
    (void)oroMemset(tpb_input, 0, config.template_size);

    size_t scratch_bytes = (size_t)result.batch_size * config.scratch_per_hash;
    uint64_t *tpb_scratch = nullptr;
    (void)oro_safe_malloc((oroDeviceptr *)&tpb_scratch, scratch_bytes);
    if (!tpb_scratch)
    {
      (void)oro_safe_free((oroDeviceptr)tpb_input);
      return false;
    }

    TuneOutputBuffer out(device_id);
    out.printf("[AUTOTUNE] GPU %d: S1 TPB sweep (%u hashes)\n", device_id, result.batch_size);

    float best_ms = 1e30f;
    uint32_t scratch_offset = 0;
    uint64_t nonce_start = 0;

    for (int tpb = 32; tpb <= 256; tpb += 32)
    {
      size_t smem = cooperative ? ((size_t)tpb * 176) : 0;
      int blks = (result.batch_size + tpb - 1) / tpb;

      void *args[] = {
          (void *)&tpb_input,
          (void *)&tpb_scratch,
          (void *)&nonce_start,
          (void *)&result.batch_size,
          (void *)&scratch_offset};

      // Probe launch + sync to catch both config and runtime errors
      oroError_t lerr = oro_safe_launch(it->second,
                                         blks, 1, 1, tpb, 1, 1,
                                         smem, stream, args, nullptr);
      if (lerr != oroSuccess) continue;
      oroError_t serr = oro_safe_stream_sync(stream);
      if (serr != oroSuccess) continue;

      // Warmup
      (void)oro_safe_launch(it->second,
                            blks, 1, 1, tpb, 1, 1,
                            smem, stream, args, nullptr);
      if (oro_safe_stream_sync(stream) != oroSuccess) continue;

      // Timed run
      oroEvent_t ev0, ev1;
      (void)oroEventCreate(&ev0);
      (void)oroEventCreate(&ev1);

      (void)oro_safe_event_record(ev0, stream);
      (void)oro_safe_launch(it->second,
                            blks, 1, 1, tpb, 1, 1,
                            smem, stream, args, nullptr);
      (void)oro_safe_event_record(ev1, stream);
      serr = oro_safe_event_sync(ev1);

      float ms = 0;
      (void)oro_safe_event_elapsed(&ms, ev0, ev1);
      (void)oroEventDestroy(ev0);
      (void)oroEventDestroy(ev1);

      if (serr != oroSuccess || ms < 0.001f) continue;

      float kh = (float)result.batch_size / (ms * 1e-3f) / 1e3f;
      out.printf("[AUTOTUNE] GPU %d:   TPB=%4d  %.2fms  %.0f kH/s%s\n",
                 device_id, tpb, ms, kh, (ms < best_ms) ? " *" : "");

      if (ms < best_ms)
      {
        best_ms = ms;
        best_s1_tpb = tpb;
      }
    }

    result.tune_keys["s1_tpb"] = (int64_t)best_s1_tpb;
    out.printf("[AUTOTUNE] GPU %d: S1 best TPB=%d\n", device_id, best_s1_tpb);
    out.flush();

    (void)oro_safe_free((oroDeviceptr)tpb_scratch);
    (void)oro_safe_free((oroDeviceptr)tpb_input);
  }

  // ── S1 bandwidth knee sweep (using winning TPB) ──
  int block_size = best_s1_tpb;
  size_t shared_mem = cooperative ? ((size_t)best_s1_tpb * 176) : 0;

  // Allocate minimal probe buffers (stage1 only needs input + scratch)
  uint8_t *probe_input = nullptr;
  (void)oro_safe_malloc((oroDeviceptr *)&probe_input, config.template_size);
  if (!probe_input)
    return false;
  (void)oroMemset(probe_input, 0, config.template_size);

  // Build probe batch sizes: from CU×block up to winning batch, ~1.5x steps
  std::vector<uint32_t> probe_batches;
  uint32_t start = (uint32_t)(compute_units * block_size);
  if (start < (uint32_t)block_size)
    start = block_size;
  for (uint32_t b = start; b <= result.batch_size;)
  {
    uint32_t aligned = (b / block_size) * block_size;
    if (aligned > 0 && (probe_batches.empty() || aligned > probe_batches.back()))
      probe_batches.push_back(aligned);
    if (b == result.batch_size)
      break;
    b = std::min((uint32_t)(b * 1.5), result.batch_size);
  }

  if (probe_batches.size() < 3)
  {
    // Too few points to detect a knee — skip S1 sweep but still run b3
    (void)oro_safe_free((oroDeviceptr)probe_input);
    TuneOutputBuffer out(device_id);
    out.printf("[AUTOTUNE] GPU %d: S1 knee sweep skipped (only %zu points, need 3+)\n",
               device_id, probe_batches.size());
    goto blake3_sweep;
  }

  {
  TuneOutputBuffer out(device_id);
  out.printf("[AUTOTUNE] GPU %d: S1 bandwidth knee sweep (TPB=%d, %zu points, %u..%u hashes)\n",
             device_id, best_s1_tpb, probe_batches.size(), probe_batches.front(), probe_batches.back());

  double peak_throughput = 0;
  uint32_t peak_batch = 0;
  std::vector<std::pair<uint32_t, double>> measurements;

  for (uint32_t probe_batch : probe_batches)
  {
    size_t scratch_bytes = (size_t)probe_batch * config.scratch_per_hash;

    uint64_t *probe_scratch = nullptr;
    (void)oro_safe_malloc((oroDeviceptr *)&probe_scratch, scratch_bytes);
    if (!probe_scratch)
      break;

    int num_blocks = (probe_batch + block_size - 1) / block_size;
    uint32_t scratch_offset = 0;
    uint64_t nonce_start = 0;

    void *args[] = {
        (void *)&probe_input,
        (void *)&probe_scratch,
        (void *)&nonce_start,
        (void *)&probe_batch,
        (void *)&scratch_offset};

    // Warmup
    (void)oro_safe_launch(it->second,
                          num_blocks, 1, 1, block_size, 1, 1,
                          shared_mem, stream, args, nullptr);
    (void)oro_safe_stream_sync(stream);

    // Timed run
    oroEvent_t ev_start, ev_stop;
    (void)oroEventCreate(&ev_start);
    (void)oroEventCreate(&ev_stop);

    (void)oro_safe_event_record(ev_start, stream);
    (void)oro_safe_launch(it->second,
                          num_blocks, 1, 1, block_size, 1, 1,
                          shared_mem, stream, args, nullptr);
    (void)oro_safe_event_record(ev_stop, stream);
    (void)oro_safe_event_sync(ev_stop);

    float ms = 0;
    (void)oro_safe_event_elapsed(&ms, ev_start, ev_stop);

    (void)oroEventDestroy(ev_start);
    (void)oroEventDestroy(ev_stop);
    (void)oro_safe_free((oroDeviceptr)probe_scratch);

    double throughput = (ms > 0.001f) ? (probe_batch / (double)ms) : 0;
    measurements.push_back({probe_batch, throughput});

    out.printf("[AUTOTUNE] GPU %d:   batch=%6u  %.2fms  %.0f H/ms\n",
               device_id, probe_batch, ms, throughput);

    if (throughput > peak_throughput)
    {
      peak_throughput = throughput;
      peak_batch = probe_batch;
    }
  }

  (void)oro_safe_free((oroDeviceptr)probe_input);

  // Two-pass knee detection: find peak, then scan forward for >10% drop
  if (peak_throughput > 0)
  {
    size_t peak_idx = 0;
    for (size_t i = 0; i < measurements.size(); i++)
    {
      if (measurements[i].second >= peak_throughput * 0.999)
      {
        peak_idx = i;
        break;
      }
    }

    uint32_t knee_batch = 0;
    for (size_t i = peak_idx + 1; i < measurements.size(); i++)
    {
      if (measurements[i].second < peak_throughput * 0.90)
      {
        knee_batch = measurements[i - 1].first;
        break;
      }
    }

    if (knee_batch > 0 && knee_batch < result.batch_size)
    {
      // Estimate split-batch s1 time vs full-batch s1 time.
      // knee throughput = peak_throughput (H/ms at the knee)
      // full-batch throughput = last measurement point
      double full_throughput = measurements.back().second;

      // Split-batch: ceil(batch/knee) chunks, each at peak throughput
      int n_chunks = ((int)result.batch_size + (int)knee_batch - 1) / (int)knee_batch;
      double split_time = (double)result.batch_size / peak_throughput; // best case
      double full_time = (double)result.batch_size / full_throughput;

      double time_delta_pct = 100.0 * (full_time - split_time) / full_time;
      double batch_ratio = (double)result.batch_size / (double)knee_batch;

      out.printf("[AUTOTUNE] GPU %d: S1 knee at %u hashes (peak %.0f H/ms, full-batch %.0f H/ms)\n",
                 device_id, knee_batch, peak_throughput, full_throughput);
      out.printf("[AUTOTUNE] GPU %d:   split estimate: %d chunks, %.1f%% faster, ratio %.1fx\n",
                 device_id, n_chunks, time_delta_pct, batch_ratio);

      // Only apply if the split is >10% faster OR the knee is less than half the batch
      if (time_delta_pct > 10.0 || batch_ratio > 2.0)
      {
        result.tune_keys["s1_knee_batch"] = (int64_t)knee_batch;
        out.printf("[AUTOTUNE] GPU %d:   -> APPLYING knee (threshold met)\n", device_id);
      }
      else
      {
        out.printf("[AUTOTUNE] GPU %d:   -> SKIPPING knee (delta %.1f%% <= 10%%, ratio %.1fx <= 2x)\n",
                   device_id, time_delta_pct, batch_ratio);
      }
    }
    else
    {
      out.printf("[AUTOTUNE] GPU %d: No S1 bandwidth knee detected (peak %.0f H/ms at %u)\n",
                 device_id, peak_throughput, peak_batch);
    }
  }

  } // end S1 knee scope
  } // end has_separate_s1

blake3_sweep:
  if (!has_separate_b3)
    return true;
  // ── Blake3 kernel sweep ──
  // Benchmarks warp-coop (variable TPB) and smem-pipelined (fixed 512 TPB).
  // Picks overall fastest and stores b3_smem (0=warp-coop, 1=smem) + b3_tpb.
  {
    auto it_wc = kernels.find("xelis_blake3_warp_coop_batch");
    auto it_smem = kernels.find("xelis_blake3_smem_batch");

    if (it_wc != kernels.end() || it_smem != kernels.end())
    {
      uint32_t b3_batch = (result.batch_size > 0)
          ? result.batch_size
          : (uint32_t)(compute_units * 256);
      size_t b3_mem = (size_t)b3_batch * config.scratch_per_hash;

      uint64_t *b3_scratch = nullptr;
      (void)oro_safe_malloc((oroDeviceptr *)&b3_scratch, b3_mem);

      uint8_t *b3_outputs = nullptr;
      (void)oro_safe_malloc((oroDeviceptr *)&b3_outputs, (size_t)b3_batch * config.hash_size);

      if (b3_scratch && b3_outputs)
      {
        (void)oroMemset(b3_scratch, 0xAB, b3_mem);
        (void)oroMemset(b3_outputs, 0, (size_t)b3_batch * config.hash_size);

        uint32_t scratch_offset = 0;
        uint64_t nonce_start = 0;
        uint64_t dummy_diff[4] = {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
                                   0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
        uint64_t *d_diff = nullptr, *d_sol = nullptr;
        (void)oro_safe_malloc((oroDeviceptr *)&d_diff, sizeof(dummy_diff));
        (void)oro_safe_malloc((oroDeviceptr *)&d_sol, sizeof(uint64_t) * 64);
        if (d_diff) (void)oroMemcpyHtoD((oroDeviceptr)d_diff, dummy_diff, sizeof(dummy_diff));
        if (d_sol)  (void)oroMemset(d_sol, 0, sizeof(uint64_t) * 64);

        void *args[] = {
            (void *)&b3_scratch,
            (void *)&b3_outputs,
            (void *)&b3_batch,
            (void *)&scratch_offset,
            (void *)&d_diff,
            (void *)&d_sol,
            (void *)&nonce_start};

        TuneOutputBuffer out(device_id);
        out.printf("[AUTOTUNE] GPU %d: Blake3 sweep (%u hashes)\n",
                   device_id, b3_batch);

        int best_tpb = xelis_b3_coop_tpb(device_id);
        float best_ms = 1e9f;
        bool best_is_smem = false;

        // Lambda to time a single blake3 launch
        auto time_b3 = [&](oroFunction_t func, int grid, int tpb, size_t smem) -> float {
          // Warmup
          (void)oro_safe_launch(func, grid, 1, 1, tpb, 1, 1,
                                smem, stream, args, nullptr);
          (void)oro_safe_stream_sync(stream);

          oroEvent_t ev_start, ev_stop;
          (void)oroEventCreate(&ev_start);
          (void)oroEventCreate(&ev_stop);

          (void)oro_safe_event_record(ev_start, stream);
          (void)oro_safe_launch(func, grid, 1, 1, tpb, 1, 1,
                                smem, stream, args, nullptr);
          (void)oro_safe_event_record(ev_stop, stream);
          (void)oro_safe_event_sync(ev_stop);

          float ms = 0;
          (void)oro_safe_event_elapsed(&ms, ev_start, ev_stop);
          (void)oroEventDestroy(ev_start);
          (void)oroEventDestroy(ev_stop);
          return ms;
        };

        // Sweep warp-coop at various TPBs (1 block per hash, no smem)
        if (it_wc != kernels.end())
        {
          out.printf("[AUTOTUNE] GPU %d:   ── warp-coop (variable TPB) ──\n", device_id);
          for (int tpb = 32; tpb <= 512; tpb += 32)
          {
            float ms = time_b3(it_wc->second, b3_batch, tpb, 0);
            float kh = (ms > 0.001f) ? (float)b3_batch / (ms * 1e-3f) / 1e3f : 0;
            out.printf("[AUTOTUNE] GPU %d:   TPB=%4d  %.2fms  %.0f kH/s%s\n",
                       device_id, tpb, ms, kh, (ms < best_ms) ? " *" : "");

            if (ms < best_ms)
            {
              best_ms = ms;
              best_tpb = tpb;
              best_is_smem = false;
            }
          }
        }

        // Sweep smem-pipelined at multiples of 32 (1 block per hash)
        if (it_smem != kernels.end())
        {
          out.printf("[AUTOTUNE] GPU %d:   ── smem pipelined (variable TPB) ──\n", device_id);
          for (int tpb = 32; tpb <= 512; tpb += 32)
          {
            oroError_t lerr = oro_safe_launch(it_smem->second, b3_batch, 1, 1,
                                               tpb, 1, 1, 0, stream, args, nullptr);
            (void)oro_safe_stream_sync(stream);
            if (lerr != oroSuccess) continue; // skip invalid TPBs

            float ms = time_b3(it_smem->second, b3_batch, tpb, 0);
            float kh = (ms > 0.001f) ? (float)b3_batch / (ms * 1e-3f) / 1e3f : 0;
            out.printf("[AUTOTUNE] GPU %d:   smem %4d  %.2fms  %.0f kH/s%s\n",
                       device_id, tpb, ms, kh, (ms < best_ms) ? " *" : "");

            if (ms < best_ms)
            {
              best_ms = ms;
              best_tpb = tpb;
              best_is_smem = true;
            }
          }
        }

        result.tune_keys["b3_tpb"] = (int64_t)best_tpb;
        result.tune_keys["b3_smem"] = (int64_t)(best_is_smem ? 1 : 0);
        float best_kh = (best_ms > 0.001f) ? (float)b3_batch / (best_ms * 1e-3f) / 1e3f : 0;
        out.printf("[AUTOTUNE] GPU %d: Blake3 best: %s TPB=%d (%.0f kH/s)\n",
                   device_id, best_is_smem ? "smem" : "warp-coop", best_tpb, best_kh);

        if (d_diff) (void)oro_safe_free((oroDeviceptr)d_diff);
        if (d_sol)  (void)oro_safe_free((oroDeviceptr)d_sol);
      }

      if (b3_scratch) (void)oro_safe_free((oroDeviceptr)b3_scratch);
      if (b3_outputs) (void)oro_safe_free((oroDeviceptr)b3_outputs);
    }
  }

  return true;
}

// Newton reciprocal LUT — lazily allocated per device, lives for process lifetime
static inline const double *xelis_get_newton_lut()
{
  static std::mutex mtx;
  static std::unordered_map<int, double *> lut_map;

  int dev = 0;
  (void)oroGetDevice(&dev);

  std::lock_guard<std::mutex> lock(mtx);
  auto it = lut_map.find(dev);
  if (it != lut_map.end())
    return it->second;

  double h_lut[NEWTON_LUT_N];
  for (int i = 0; i < NEWTON_LUT_N; i++)
  {
    double v_center = (double)(NEWTON_LUT_BASE + i + 0.5) * (double)(1 << NEWTON_LUT_SHIFT);
    h_lut[i] = 1.0 / v_center;
  }

  double *d_lut = nullptr;
  (void)oro_safe_malloc((oroDeviceptr *)&d_lut, sizeof(h_lut));
  if (d_lut)
  {
    (void)oroMemcpyHtoD((oroDeviceptr)d_lut, h_lut, sizeof(h_lut));
    lut_map[dev] = d_lut;
  }
  return d_lut;
}

// AMD AES texture objects — lazily created per module, lives for process lifetime
// On GCN/CDNA (HAS_TCP), creates hipTextureObject_t for sbox/GF2/GF3 and uploads
// to __device__ globals in the HIPRTC module. On RDNA (no TCP) or NVIDIA, no-op.
static inline void xelis_setup_aes_textures(oroModule_t module)
{
  if (!module) return;
  if (!tnn_is_amd_device(0)) return;  // NVIDIA uses __ldg, not textures

  static std::mutex mtx;
  static std::unordered_set<oroModule_t> initialized;

  std::lock_guard<std::mutex> lock(mtx);
  if (initialized.count(module)) return;
  initialized.insert(module);

  // Check if module has the texture symbols (only present on TCP arches)
  oroDeviceptr_t d_sym = 0;
  size_t sym_size = 0;
  if (oroModuleGetGlobal(&d_sym, &sym_size, module, "d_aes_sbox_tex") != oroSuccess)
  {
    // Clear the sticky error — RDNA doesn't have these symbols and that's expected
    (void)oroGetLastError();
    return;  // No TCP symbols compiled — RDNA path uses __constant__ directly
  }

  // Upload AES tables to device global memory for texture binding
  static const uint8_t h_sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16};
  static const uint8_t h_gf2[256] = {
    0x00,0x02,0x04,0x06,0x08,0x0a,0x0c,0x0e,0x10,0x12,0x14,0x16,0x18,0x1a,0x1c,0x1e,
    0x20,0x22,0x24,0x26,0x28,0x2a,0x2c,0x2e,0x30,0x32,0x34,0x36,0x38,0x3a,0x3c,0x3e,
    0x40,0x42,0x44,0x46,0x48,0x4a,0x4c,0x4e,0x50,0x52,0x54,0x56,0x58,0x5a,0x5c,0x5e,
    0x60,0x62,0x64,0x66,0x68,0x6a,0x6c,0x6e,0x70,0x72,0x74,0x76,0x78,0x7a,0x7c,0x7e,
    0x80,0x82,0x84,0x86,0x88,0x8a,0x8c,0x8e,0x90,0x92,0x94,0x96,0x98,0x9a,0x9c,0x9e,
    0xa0,0xa2,0xa4,0xa6,0xa8,0xaa,0xac,0xae,0xb0,0xb2,0xb4,0xb6,0xb8,0xba,0xbc,0xbe,
    0xc0,0xc2,0xc4,0xc6,0xc8,0xca,0xcc,0xce,0xd0,0xd2,0xd4,0xd6,0xd8,0xda,0xdc,0xde,
    0xe0,0xe2,0xe4,0xe6,0xe8,0xea,0xec,0xee,0xf0,0xf2,0xf4,0xf6,0xf8,0xfa,0xfc,0xfe,
    0x1b,0x19,0x1f,0x1d,0x13,0x11,0x17,0x15,0x0b,0x09,0x0f,0x0d,0x03,0x01,0x07,0x05,
    0x3b,0x39,0x3f,0x3d,0x33,0x31,0x37,0x35,0x2b,0x29,0x2f,0x2d,0x23,0x21,0x27,0x25,
    0x5b,0x59,0x5f,0x5d,0x53,0x51,0x57,0x55,0x4b,0x49,0x4f,0x4d,0x43,0x41,0x47,0x45,
    0x7b,0x79,0x7f,0x7d,0x73,0x71,0x77,0x75,0x6b,0x69,0x6f,0x6d,0x63,0x61,0x67,0x65,
    0x9b,0x99,0x9f,0x9d,0x93,0x91,0x97,0x95,0x8b,0x89,0x8f,0x8d,0x83,0x81,0x87,0x85,
    0xbb,0xb9,0xbf,0xbd,0xb3,0xb1,0xb7,0xb5,0xab,0xa9,0xaf,0xad,0xa3,0xa1,0xa7,0xa5,
    0xdb,0xd9,0xdf,0xdd,0xd3,0xd1,0xd7,0xd5,0xcb,0xc9,0xcf,0xcd,0xc3,0xc1,0xc7,0xc5,
    0xfb,0xf9,0xff,0xfd,0xf3,0xf1,0xf7,0xf5,0xeb,0xe9,0xef,0xed,0xe3,0xe1,0xe7,0xe5};
  static const uint8_t h_gf3[256] = {
    0x00,0x03,0x06,0x05,0x0c,0x0f,0x0a,0x09,0x18,0x1b,0x1e,0x1d,0x14,0x17,0x12,0x11,
    0x30,0x33,0x36,0x35,0x3c,0x3f,0x3a,0x39,0x28,0x2b,0x2e,0x2d,0x24,0x27,0x22,0x21,
    0x60,0x63,0x66,0x65,0x6c,0x6f,0x6a,0x69,0x78,0x7b,0x7e,0x7d,0x74,0x77,0x72,0x71,
    0x50,0x53,0x56,0x55,0x5c,0x5f,0x5a,0x59,0x48,0x4b,0x4e,0x4d,0x44,0x47,0x42,0x41,
    0xc0,0xc3,0xc6,0xc5,0xcc,0xcf,0xca,0xc9,0xd8,0xdb,0xde,0xdd,0xd4,0xd7,0xd2,0xd1,
    0xf0,0xf3,0xf6,0xf5,0xfc,0xff,0xfa,0xf9,0xe8,0xeb,0xee,0xed,0xe4,0xe7,0xe2,0xe1,
    0xa0,0xa3,0xa6,0xa5,0xac,0xaf,0xaa,0xa9,0xb8,0xbb,0xbe,0xbd,0xb4,0xb7,0xb2,0xb1,
    0x90,0x93,0x96,0x95,0x9c,0x9f,0x9a,0x99,0x88,0x8b,0x8e,0x8d,0x84,0x87,0x82,0x81,
    0x9b,0x98,0x9d,0x9e,0x97,0x94,0x91,0x92,0x83,0x80,0x85,0x86,0x8f,0x8c,0x89,0x8a,
    0xab,0xa8,0xad,0xae,0xa7,0xa4,0xa1,0xa2,0xb3,0xb0,0xb5,0xb6,0xbf,0xbc,0xb9,0xba,
    0xfb,0xf8,0xfd,0xfe,0xf7,0xf4,0xf1,0xf2,0xe3,0xe0,0xe5,0xe6,0xef,0xec,0xe9,0xea,
    0xcb,0xc8,0xcd,0xce,0xc7,0xc4,0xc1,0xc2,0xd3,0xd0,0xd5,0xd6,0xdf,0xdc,0xd9,0xda,
    0x5b,0x58,0x5d,0x5e,0x57,0x54,0x51,0x52,0x43,0x40,0x45,0x46,0x4f,0x4c,0x49,0x4a,
    0x6b,0x68,0x6d,0x6e,0x67,0x64,0x61,0x62,0x73,0x70,0x75,0x76,0x7f,0x7c,0x79,0x7a,
    0x3b,0x38,0x3d,0x3e,0x37,0x34,0x31,0x32,0x23,0x20,0x25,0x26,0x2f,0x2c,0x29,0x2a,
    0x0b,0x08,0x0d,0x0e,0x07,0x04,0x01,0x02,0x13,0x10,0x15,0x16,0x1f,0x1c,0x19,0x1a};

  auto make_tex_u8 = [](const uint8_t *h_data, size_t bytes) -> oroTextureObject_t {
    uint8_t *d_buf = nullptr;
    (void)oro_safe_malloc((oroDeviceptr *)&d_buf, bytes);
    if (!d_buf) return 0;
    (void)oroMemcpyHtoD((oroDeviceptr)d_buf, h_data, bytes);

    oroResourceDesc rd = {};
    rd.resType = oroResourceTypeLinear;
    rd.res.linear.devPtr = d_buf;
    rd.res.linear.desc = oroCreateChannelDesc(8, 0, 0, 0, oroChannelFormatKindUnsigned);
    rd.res.linear.sizeInBytes = bytes;
    oroTextureDesc td = {};
    td.readMode = oroReadModeElementType;
    oroTextureObject_t tex = 0;
    (void)oroCreateTextureObject(&tex, &rd, &td, nullptr);
    return tex;
  };

  oroTextureObject_t sbox_tex = make_tex_u8(h_sbox, 256);
  oroTextureObject_t gf2_tex  = make_tex_u8(h_gf2, 256);
  oroTextureObject_t gf3_tex  = make_tex_u8(h_gf3, 256);

  // Write texture handles to __device__ globals in the HIPRTC module
  auto set_global = [&](const char *name, const void *data, size_t size) {
    oroDeviceptr_t dptr = 0;
    size_t sz = 0;
    if (oroModuleGetGlobal(&dptr, &sz, module, name) == oroSuccess && sz >= size)
      (void)oroMemcpyHtoD(dptr, data, size);
  };
  set_global("d_aes_sbox_tex", &sbox_tex, sizeof(sbox_tex));
  set_global("d_aes_gf2_tex",  &gf2_tex,  sizeof(gf2_tex));
  set_global("d_aes_gf3_tex",  &gf3_tex,  sizeof(gf3_tex));

  TNN_LOG_DEBUG("[XELIS] AES textures initialized (sbox=%llu gf2=%llu gf3=%llu)\n",
    (unsigned long long)sbox_tex, (unsigned long long)gf2_tex, (unsigned long long)gf3_tex);
}

// Bottleneck setup: runs s1 to fill scratchpad with realistic data before s3 timing.
// Op dispatch, branch patterns, and memory indices all depend on scratch content.
// Syncs stream so s1 is complete before the timer starts.
inline bool xelis_v3_bottleneck_setup(
    const KernelMap &kernels,
    const KernelLaunchContext &ctx)
{
  auto strat = static_cast<XelisStrategy>(ctx.strategy);
  if (strat != XelisStrategy::Sep && strat != XelisStrategy::SepP)
    return true; // non-Sep: no setup needed (full pipeline runs via execute_fn)

  int dev = 0;
  (void)oroGetDevice(&dev);

  if (!xelis_launch_stage1(kernels, ctx, dev))
    return false;
  (void)oro_safe_stream_sync(ctx.stream);
  return true;
}

// Bottleneck-only execute for autotune: launches only s3.
// Scratchpad already filled by bottleneck_setup_fn (s1) before timer started.
// For non-Sep strategies, returns false so autotune falls through to execute_fn.
inline bool xelis_v3_bottleneck_execute(
    const KernelMap &kernels,
    const KernelLaunchContext &ctx)
{
  auto strat = static_cast<XelisStrategy>(ctx.strategy);
  if (strat != XelisStrategy::Sep && strat != XelisStrategy::SepP)
    return false; // non-Sep: autotune will use execute_fn instead

  const double *newton_lut = xelis_get_newton_lut();

  auto it = kernels.find("xelis_s3_hybrid_v2_noblake_kernel");
  if (it == kernels.end())
    return false;

  uint32_t scratch_offset = 0;
  void *args[] = {
      (void *)&ctx.d_scratch,
      (void *)&ctx.batch_size,
      (void *)&scratch_offset,
      (void *)&ctx.d_difficulty_target,
      (void *)&ctx.d_solutions,
      (void *)&ctx.nonce_start,
      (void *)&newton_lut};

  oroError_t err = oro_safe_launch(
      it->second,
      ctx.num_blocks, 1, 1,
      ctx.block_size, 1, 1,
      0, ctx.stream,
      args, nullptr);
  if (err != oroSuccess)
  {
    TNN_LOG_ERROR("[XELIS] s3 bottleneck launch failed: %s\n", tnn_error_string(err));
    return false;
  }
  return true;
}

inline bool xelis_v3_execute(
    const KernelMap &kernels,
    const KernelLaunchContext &ctx)
{
  int dev = 0;
  (void)oroGetDevice(&dev);

  const double *newton_lut = xelis_get_newton_lut();
  xelis_setup_aes_textures(ctx.module);

  const auto strategy = static_cast<XelisStrategy>(ctx.strategy);
  uint32_t scratch_offset = 0;

  switch (strategy)
  {

  case XelisStrategy::Mono:
  {
    auto it = kernels.find("xelis_hash_v3_kernel");
    if (it == kernels.end())
      return false;

    void *args[] = {
        (void *)&ctx.d_input,
        (void *)&ctx.d_outputs,
        (void *)&ctx.d_scratch,
        (void *)&ctx.nonce_start,
        (void *)&ctx.batch_size,
        (void *)&ctx.d_difficulty_target,
        (void *)&ctx.d_solutions,
        (void *)&newton_lut};

    oroError_t err = oro_safe_launch(
        it->second,
        ctx.num_blocks, 1, 1,
        ctx.block_size, 1, 1,
        0, ctx.stream,  // no LDS sbox — AES via __ldg / col1
        args, nullptr);
    if (err != oroSuccess)
    {
      TNN_LOG_ERROR("[XELIS] xelis_hash_v3_kernel launch failed: %s\n", tnn_error_string(err));
      return false;
    }
    return true;
  }

  case XelisStrategy::Baseline:
  {
    // s1+s3 fused, then blake3 separate
    auto it = kernels.find("xelis_s13_noblake_kernel");
    if (it == kernels.end())
      return default_monolithic_execute(kernels, ctx);

    void *args[] = {
        (void *)&ctx.d_input,
        (void *)&ctx.d_scratch,
        (void *)&ctx.nonce_start,
        (void *)&ctx.batch_size,
        (void *)&scratch_offset,
        (void *)&newton_lut};

    oroError_t err = oro_safe_launch(
        it->second,
        ctx.num_blocks, 1, 1,
        ctx.block_size, 1, 1,
        0, ctx.stream,
        args, nullptr);
    if (err != oroSuccess)
    {
      TNN_LOG_ERROR("[XELIS] s13_noblake launch failed: %s\n", tnn_error_string(err));
      return false;
    }
    return xelis_launch_blake3(kernels, ctx, dev);
  }

  case XelisStrategy::Sep:
  {
    // stage1 separate, s3 separate, blake3 separate
    if (!xelis_launch_stage1(kernels, ctx, dev))
      return false;

    auto it = kernels.find("xelis_s3_hybrid_v2_noblake_kernel");
    if (it == kernels.end())
      return false;

    void *args[] = {
        (void *)&ctx.d_scratch,
        (void *)&ctx.batch_size,
        (void *)&scratch_offset,
        (void *)&ctx.d_difficulty_target,
        (void *)&ctx.d_solutions,
        (void *)&ctx.nonce_start,
        (void *)&newton_lut};

    oroError_t err = oro_safe_launch(
        it->second,
        ctx.num_blocks, 1, 1,
        ctx.block_size, 1, 1,
        0, ctx.stream,
        args, nullptr);
    if (err != oroSuccess)
    {
      TNN_LOG_ERROR("[XELIS] s3_hybrid_v2_noblake launch failed: %s\n", tnn_error_string(err));
      return false;
    }
    // Dispatch blake3: smem pipelined if tuned, else warp-coop with tuned TPB
    if (ctx.get_tune_key("b3_smem", 0))
      return xelis_launch_blake3_smem(kernels, ctx, dev);
    return xelis_launch_blake3(kernels, ctx, dev);
  }

  case XelisStrategy::Neo:
  {
    // stage1 separate, s3+b3 fused
    if (!xelis_launch_stage1(kernels, ctx, dev))
      return false;

    auto it = kernels.find("xelis_s3b3_hybrid_v2_kernel");
    if (it == kernels.end())
      return false;

    void *args[] = {
        (void *)&ctx.d_scratch,
        (void *)&ctx.d_outputs,
        (void *)&ctx.batch_size,
        (void *)&scratch_offset,
        (void *)&ctx.d_difficulty_target,
        (void *)&ctx.d_solutions,
        (void *)&ctx.nonce_start,
        (void *)&newton_lut};

    TNN_LOG_TRACE("[LAUNCH] Neo s3b3: kernel=%p, grid=%d, block=%d\n",
                  (void *)it->second, ctx.num_blocks, ctx.block_size);
    fflush(stdout);

    oroError_t err = oro_safe_launch(
        it->second,
        ctx.num_blocks, 1, 1,
        ctx.block_size, 1, 1,
        0, ctx.stream,
        args, nullptr);

    TNN_LOG_TRACE("[LAUNCH] Neo s3b3 returned %d (%s)\n", (int)err, tnn_error_string(err));
    fflush(stdout);

    if (err != oroSuccess)
    {
      TNN_LOG_ERROR("[XELIS] s3b3_hybrid_v2 launch failed: %s\n", tnn_error_string(err));
      return false;
    }
    return true;
  }

  case XelisStrategy::SepP:
  {
    // stage1 separate, s3 separate, blake3 smem pipelined (512 TPB)
    if (!xelis_launch_stage1(kernels, ctx, dev))
      return false;

    auto it = kernels.find("xelis_s3_hybrid_v2_noblake_kernel");
    if (it == kernels.end())
      return false;

    void *args[] = {
        (void *)&ctx.d_scratch,
        (void *)&ctx.batch_size,
        (void *)&scratch_offset,
        (void *)&ctx.d_difficulty_target,
        (void *)&ctx.d_solutions,
        (void *)&ctx.nonce_start,
        (void *)&newton_lut};

    oroError_t err = oro_safe_launch(
        it->second,
        ctx.num_blocks, 1, 1,
        ctx.block_size, 1, 1,
        0, ctx.stream,
        args, nullptr);
    if (err != oroSuccess)
    {
      TNN_LOG_ERROR("[XELIS] s3_hybrid_v2_noblake launch failed: %s\n", tnn_error_string(err));
      return false;
    }
    return xelis_launch_blake3_smem(kernels, ctx, dev);
  }

  default:
    TNN_LOG_ERROR("[XELIS] Unknown strategy %d\n", (int)strategy);
    return false;
  }
}

// ============================================================================
// Xelis V3 Configuration
// ============================================================================
inline AlgoConfig XELIS_V3_CONFIG = {
    .name = "xelis_v3",
    .source_path = "src/tnn_hip/crypto/xelis-hash/xelis-hash-v3.hip",
#ifdef TNN_XELISHASH
    .source = hip_xelis_v3_source::SRC_TNN_HIP_CRYPTO_XELIS_HASH_XELIS_HASH_V3_HIP_SOURCE.data(),
#else
    .source = {},
#endif

    // All kernels used across strategies
    .kernel_names = {
        "xelis_hash_v3_kernel",              // Mono (primary/fallback)
        "xelis_stage1_kernel",               // Sep/Neo stage1
        "xelis_stage1_cooperative",          // Sep/Neo stage1 (RDNA+/Ampere+)
        "xelis_s13_noblake_kernel",          // Baseline (s1+s3 fused)
        "xelis_s3_hybrid_v2_noblake_kernel", // Sep (s3 only)
        "xelis_s3b3_hybrid_v2_kernel",       // Neo (s3+b3 fused)
        "xelis_blake3_batch",                // Baseline/Sep blake3 (original)
        "xelis_blake3_opt_batch",            // Sep blake3 (optimized: u32 loads, branched merge, shared cv_stack)
        "xelis_blake3_warp_coop_batch",      // Sep blake3 (warp-cooperative: global-CV, no smem)
        "xelis_blake3_smem_batch"            // SepP blake3 (smem pipelined: 512 TPB, smem tree)
    },

    .kernel_name = "",

#ifdef TNN_XELISHASH
    .rtc_headers = build_rtc_headers(hip_embedded::XELIS_SOURCES, hip_embedded::COMMON_HEADERS),
#else
    .rtc_headers = {},
#endif
    .template_size = 112,
    .hash_size = 32,
    .nonce_size = 8,
    .scratch_per_hash = (531 * 128 + 1) * sizeof(uint64_t) + 8512, // +1 for nonce storage, +8512 for blake3 global-CV workspace
    .preferred_block_size = 32,
    .algo_id = ALGO_XELISV3,
    .calc_shared_mem = xelis_v3_shared_mem,

    .category = AlgoCategory::MemoryHard,
    .enable_reg_tuning = true,

    .amd_blocks = {32, 256, 32},
    .nvidia_blocks = {32, 256, 32},
    .target_batch_time_ms = 1250.0,
    .max_batch_time_ms = 2000.0,
    .min_batch_time_ms = 100.0,
    .enable_autotune = true,
    .autotune_warmup = 1,
    .autotune_iterations = 1,
    .memory_reserve_mb = 32.0,
    .memory_usage_factor = 1.0,
    .batch_step_denom = 1,

    .execute_fn = xelis_v3_execute,

    // Sep — s3 tuned in isolation, s1+b3 tuned as edge kernels
    .strategy_variants = {// (uint8_t)XelisStrategy::Mono,
                          // (uint8_t)XelisStrategy::Baseline,
                          (uint8_t)XelisStrategy::Sep,
                          // (uint8_t)XelisStrategy::Neo,
                          // (uint8_t)XelisStrategy::SepP,
                          },
    .strategy_names = {/*"Mono", "Baseline",*/ "Sep", /*"Neo", "SepP"*/},

    // Bottleneck kernel per strategy (for occupancy queries) — must match strategy_variants order
    .strategy_bottleneck_kernels = {"xelis_s3_hybrid_v2_noblake_kernel"},

    .occupancy_threshold = 0.66,

    .bottleneck_execute_fn = xelis_v3_bottleneck_execute,
    .bottleneck_setup_fn = xelis_v3_bottleneck_setup,

    .tune_key_probe_fn = xelis_tune_key_probe,
};

// ============================================================================
// KawPow Configuration
// ============================================================================
#ifdef TNN_KAWPOW

// Number of periods to compile for benchmarking (measures average across programs)
static constexpr int KAWPOW_BENCH_PERIODS = 1;
// Base block number for bench programs — approximate current RVN chain height.
// Avoids degenerate period-0 seed (all-zero FNV input). Period = block/3.
static constexpr int KAWPOW_BENCH_BASE_BLOCK = 7500;

// Override block height for --bench-kawpow <block_height>.  -1 = use default.
// Uses function-local static to avoid ODR issues across translation units.
inline int& kawpow_bench_block_override() {
    static int val = -1;
    return val;
}

inline int kawpow_bench_block() {
    int v = kawpow_bench_block_override();
    return (v >= 0) ? v : KAWPOW_BENCH_BASE_BLOCK;
}

enum class KawPowStrategy : uint8_t {
    Mono        = 0,  // Single monolithic kernel (seed keccak + progpow + final keccak)
    Split2Way   = 4,  // 3 kernels: seed → progpow_2way → final (2 hashes per group)
    Split4Way   = 5,
};

// Per-device KawPow state (DAG, epoch info)
// Per-mode (regular / dev) live state — DAG + compiled kernels.
// Two instances live inside KawPowAlgoData so dev and regular can
// track independent epochs/periods without ping-ponging rebuilds.
struct KawPowLiveState {
    AlignedDevAlloc dag_alloc{};
    AlignedDevAlloc l1_alloc{};

    uint32_t* d_dag = nullptr;
    uint32_t* d_l1_cache = nullptr;

    uint32_t  dag_num_items = 0;
    size_t    dag_size_bytes = 0;

    // Barrett reduction for dag_addr %= dag_num_items
    uint32_t  barrett_m = 0;
    uint32_t  barrett_shift = 0;

    int current_epoch = -1;
    int current_period = -1;

    oroModule_t   live_module = nullptr;
    oroFunction_t live_mono_kernel = nullptr;
    oroFunction_t live_seed_kernel = nullptr;
    oroFunction_t live_progpow_2way_kernel = nullptr;
    oroFunction_t live_progpow_4way_kernel = nullptr;
    oroFunction_t live_final_kernel = nullptr;

    bool alloc_dag(size_t bytes, size_t align = 256, int device_id = 0) {
        oroFreeAligned(dag_alloc);
        d_dag = nullptr;
        dag_size_bytes = 0;

        oroError_t err = oroMallocAligned(dag_alloc, bytes, align);
        if (err != oroSuccess) return false;

        d_dag = reinterpret_cast<uint32_t*>(dag_alloc.aligned);
        dag_size_bytes = bytes;

        oroMemAdvise(d_dag, dag_size_bytes, oroMemAdviseSetReadMostly, device_id);
        (void)oroGetLastError(); // non-fatal if unsupported

        return true;
    }

    bool alloc_l1_cache(size_t bytes, size_t align = 64, int device_id = 0) {
        oroFreeAligned(l1_alloc);
        d_l1_cache = nullptr;

        oroError_t err = oroMallocAligned(l1_alloc, bytes, align);
        if (err != oroSuccess) return false;

        d_l1_cache = reinterpret_cast<uint32_t*>(l1_alloc.aligned);

        oroMemAdvise(d_dag, dag_size_bytes, oroMemAdviseSetReadMostly, device_id);
        (void)oroGetLastError(); // non-fatal if unsupported

        return true;
    }

    void free_dag() {
        oroFreeAligned(dag_alloc);
        oroFreeAligned(l1_alloc);

        d_dag = nullptr;
        d_l1_cache = nullptr;
        dag_num_items = 0;
        dag_size_bytes = 0;
    }

    void free_module() {
        if (live_module) {
            oroModuleUnload(live_module);
            live_module = nullptr;
        }
        live_mono_kernel = nullptr;
        live_seed_kernel = nullptr;
        live_progpow_2way_kernel = nullptr;
        live_progpow_4way_kernel = nullptr;
        live_final_kernel = nullptr;
    }
};

struct KawPowAlgoData {
    // Two live states: [0] = regular, [1] = dev
    KawPowLiveState live[2];

    uint32_t  block_number = 0;

    // Multi-period bench: compiled kernels for periods 0..N-1
    oroModule_t   period_modules[KAWPOW_BENCH_PERIODS] = {};
    oroFunction_t period_mono_kernels[KAWPOW_BENCH_PERIODS] = {};
    oroFunction_t period_seed_kernels[KAWPOW_BENCH_PERIODS] = {};
    oroFunction_t period_progpow_2way_kernels[KAWPOW_BENCH_PERIODS] = {};
    oroFunction_t period_progpow_4way_kernels[KAWPOW_BENCH_PERIODS] = {};
    oroFunction_t period_final_kernels[KAWPOW_BENCH_PERIODS] = {};
    int           num_period_kernels = 0;

    // Split strategy: intermediate buffer (16 uint32 per hash)
    AlignedDevAlloc intermediate_alloc{};
    uint32_t* d_intermediate = nullptr;
    size_t    intermediate_capacity = 0;

    // Strategy selection
    KawPowStrategy best_strategy = KawPowStrategy::Split2Way;

    // Compilation state (saved during pre_tune for runtime recompilation)
    std::vector<std::string> compile_opts;
    int compile_device_id = 0;

    // DAG gen kernel — compiled on init thread, reloaded on context refresh.
    oroModule_t   dag_gen_module = nullptr;
    oroFunction_t dag_gen_function = nullptr;

    // Pending rebuild: set by deps_changed, consumed by main_thread_fn.
    int  pending_rebuild_epoch = -1;
    bool pending_rebuild_is_dev = false;

    // Ensure intermediate buffer can hold at least `needed` hashes
    bool ensure_intermediate(size_t needed) {
        if (d_intermediate && intermediate_capacity >= needed) return true;

        oroFreeAligned(intermediate_alloc);
        d_intermediate = nullptr;
        intermediate_capacity = 0;

        constexpr size_t kAlign = 64;  // 64 B per hash record
        const size_t bytes = needed * 16 * sizeof(uint32_t);

        oroError_t err = oroMallocAligned(intermediate_alloc, bytes, kAlign);
        if (err != oroSuccess) {
            return false;
        }

        d_intermediate = reinterpret_cast<uint32_t*>(intermediate_alloc.aligned);
        intermediate_capacity = needed;
        return true;
    }

    void free_intermediate() {
        oroFreeAligned(intermediate_alloc);
        d_intermediate = nullptr;
        intermediate_capacity = 0;
    }
};

// Shared mem for monolithic kernel: L1 cache + seed_state spill
inline size_t kawpow_shared_mem_mono(int block_size)
{
  return 16384 + (size_t)(block_size / 16) * 8 * sizeof(uint32_t);
}

// Shared mem for split progpow kernel: L1 cache only (no seed_state needed)
inline size_t kawpow_shared_mem_split(int /*block_size*/)
{
  return 16384;
}

// Default: mono (used by framework for calc_shared_mem)
inline size_t kawpow_shared_mem(int block_size)
{
  return kawpow_shared_mem_mono(block_size);
}

// ---------------------------------------------------------------------------
// Helper: launch the 3-kernel split pipeline on raw kernel functions
// seed_block/final_block are tunable via tune_keys (swept in tune_key_probe)
// ---------------------------------------------------------------------------
inline void kawpow_launch_split(
    oroFunction_t seed_fn, oroFunction_t progpow_fn, oroFunction_t final_fn,
    uint32_t* d_header, uint32_t* d_dag, uint64_t nonce_start, uint32_t batch_size,
    uint64_t* d_target, uint64_t* d_solutions,
    uint32_t* d_result_hashes, uint32_t* d_l1_cache,
    uint32_t* d_intermediate,
    int progpow_grid, int progpow_block,
    int seed_block, int final_block,
    oroStream_t stream,
    size_t progpow_shared_mem = (size_t)-1 /* -1 = use default kawpow_shared_mem_split */)
{
    if (progpow_shared_mem == (size_t)-1)
        progpow_shared_mem = kawpow_shared_mem_split(progpow_block);

    // Kernel A: seed keccak (1 thread/hash, no LDS)
    int seed_grid = ((int)batch_size + seed_block - 1) / seed_block;
    {
        void* args[] = { &d_header, &nonce_start, &batch_size, &d_intermediate };
        oroError_t e = oroModuleLaunchKernel(seed_fn, seed_grid, 1, 1, seed_block, 1, 1,
                              0, stream, args, nullptr);
        if (e != oroSuccess)
            fprintf(stderr, "[KawPow] seed launch failed: %s (grid=%d, block=%d, batch=%u)\n",
                    tnn_error_string(e), seed_grid, seed_block, batch_size);
    }

    // Kernel B: progpow main loop (Barrett constants are compile-time injected)
    {
        void* args[] = { &d_dag, &batch_size, &d_l1_cache, &d_intermediate };
        oroError_t e = oroModuleLaunchKernel(progpow_fn, progpow_grid, 1, 1, progpow_block, 1, 1,
                              progpow_shared_mem, stream, args, nullptr);
        if (e != oroSuccess)
            fprintf(stderr, "[KawPow] progpow launch failed: %s (grid=%d, block=%d, smem=%zu)\n",
                    tnn_error_string(e), progpow_grid, progpow_block, progpow_shared_mem);
    }

    // Kernel C: final keccak + target check (1 thread/hash, no LDS)
    int final_grid = ((int)batch_size + final_block - 1) / final_block;
    {
        void* args[] = { &nonce_start, &batch_size, &d_target, &d_solutions,
                         &d_result_hashes, &d_intermediate };
        oroError_t e = oroModuleLaunchKernel(final_fn, final_grid, 1, 1, final_block, 1, 1,
                              0, stream, args, nullptr);
        if (e != oroSuccess)
            fprintf(stderr, "[KawPow] final launch failed: %s (grid=%d, block=%d)\n",
                    tnn_error_string(e), final_grid, final_block);
    }
}

// Source transform: inject period-0 program + RVN coin padding
inline std::string kawpow_source_transform(const std::string& source, int device_id)
{
    (void)device_id;
    std::string src = kawpow_proggen::inject_coin_padding(source, *currentKawpowPadding);
    src = kawpow_proggen::inject_dag_constants(src, 1, 1, 0); // placeholder — real values injected in pre_tune

    static const uint32_t placeholder_l1_words[4096] = {0};
    src = kawpow_proggen::inject_constant_l1_table(src, placeholder_l1_words, 4096);

    src = kawpow_proggen::inject_program(src, kawpow_bench_block());
    return src;
}

// Pre-tune setup: generate DAG at bench epoch, upload to GPU
inline bool kawpow_pre_tune(const KernelMap& kernels, const oroDeviceProp_t& props,
                             int device_id, void** algo_data)
{
    (void)kernels; (void)props;

    // Ensure HIP runtime is initialized on this thread
    oroSetDevice(device_id);

    const int epoch = ethash::get_epoch_number(kawpow_bench_block());
    auto ctx = ethash::create_epoch_context(epoch);
    if (!ctx) {
        fprintf(stderr, "[KawPow] GPU %d: Failed to create epoch context\n", device_id);
        return false;
    }

    auto* kp = new KawPowAlgoData();
    auto& ls = kp->live[0]; // pre_tune always sets up the regular live state
    kp->block_number = 0;
    ls.current_epoch = epoch;
    ls.dag_num_items = (uint32_t)(ctx->full_dataset_num_items / 2);
    size_t dag_words = (size_t)ls.dag_num_items * 64;
    ls.dag_size_bytes = dag_words * sizeof(uint32_t);

    // Barrett reduction constants for dag_addr %= dag_num_items
    {
        uint32_t d = ls.dag_num_items;
        uint32_t s = 0;
        uint64_t m64 = (1ULL << 32) / d;
        ls.barrett_m = (uint32_t)m64;
        ls.barrett_shift = s;
    }

    printf("[KawPow] GPU %d: Generating DAG on GPU for epoch %d: %u items (%.1f GB)...\n",
           device_id, epoch, ls.dag_num_items, ls.dag_size_bytes / (1024.0 * 1024.0 * 1024.0));
    fflush(stdout);

    // ---- Allocate DAG on GPU ----
    if (!ls.alloc_dag(ls.dag_size_bytes, 1 << 20, device_id)) {
        fprintf(stderr, "[KawPow] GPU %d: DAG alloc failed (need %.1f GB)\n",
                device_id, ls.dag_size_bytes / (1024.0 * 1024.0 * 1024.0));
        delete kp;
        return false;
    }

    // ---- Upload light cache to GPU ----
    uint32_t num_cache_items = (uint32_t)ctx->light_cache_num_items;
    size_t cache_bytes = (size_t)num_cache_items * 64; // 16 uint32 per hash512
    uint32_t* d_light_cache = nullptr;
    auto err = oroMalloc((oroDeviceptr*)&d_light_cache, cache_bytes);
    if (err != oroSuccess) {
        fprintf(stderr, "[KawPow] GPU %d: Light cache alloc failed: %s\n",
                device_id, tnn_error_string(err));
        ls.free_dag();
        delete kp;
        return false;
    }
    oroMemcpy(d_light_cache, ctx->light_cache, cache_bytes, oroMemcpyHostToDevice);

    printf("[KawPow] GPU %d: Light cache uploaded (%.1f MB, %u items)\n",
           device_id, cache_bytes / (1024.0 * 1024.0), num_cache_items);
    fflush(stdout);

    // ---- Compile DAG gen kernel ----
    bool is_amd = tnn_is_amd_device(device_id);
    auto& compiler = RTCCompiler::instance();

    for (const auto& h : hip_embedded::KAWPOW_HEADERS)
        compiler.add_header_source(std::string(h.path), std::string(h.source));
    for (const auto& h : hip_embedded::COMMON_HEADERS)
        compiler.add_header_source(std::string(h.path), std::string(h.source));

    std::vector<std::string> dag_opts;
    if (is_amd) {
        dag_opts = {"-O3", "-mno-cumode", "-ffast-math"};
        if (props.gcnArchName[0] != '\0')
            dag_opts.push_back(std::string("--gpu-architecture=") + props.gcnArchName);
    } else {
        dag_opts = {"--dopt=on", "--use_fast_math"};
        char buf[32];
        std::snprintf(buf, sizeof(buf), "sm_%d%d", props.major, props.minor);
        dag_opts.push_back(std::string("--gpu-architecture=") + buf);
    }

    std::string dag_src(hip_ethash_dag_source::SRC_TNN_HIP_CRYPTO_KAWPOW_ETHASH_DAG_GEN_HIP_SOURCE);
    auto dag_ck = compiler.compile_from_source(dag_src, "ethash-dag-gen.hip",
                                                "ethash_dag_gen_kernel", dag_opts, device_id);
    if (!dag_ck.function) {
        fprintf(stderr, "[KawPow] GPU %d: DAG gen kernel compile failed\n", device_id);
        oroFree((oroDeviceptr)d_light_cache);
        ls.free_dag();
        delete kp;
        return false;
    }

    // ---- Launch DAG gen kernel ----
    auto t0 = std::chrono::steady_clock::now();

    uint32_t dag_items = ls.dag_num_items;
    int block_size = 256;
    int grid_size = (dag_items + block_size - 1) / block_size;

    void* dag_args[] = { &d_light_cache, &ls.d_dag, &num_cache_items, &dag_items };
    err = oroModuleLaunchKernel(dag_ck.function, grid_size, 1, 1, block_size, 1, 1,
                                 0, nullptr, dag_args, nullptr);
    if (err != oroSuccess) {
        fprintf(stderr, "[KawPow] GPU %d: DAG gen kernel launch failed: %s\n",
                device_id, tnn_error_string(err));
        oroFree((oroDeviceptr)d_light_cache);
        ls.free_dag();
        delete kp;
        return false;
    }
    oroDeviceSynchronize();

    auto t1 = std::chrono::steady_clock::now();
    printf("[KawPow] GPU %d: DAG generated on GPU in %.2fs\n",
           device_id, std::chrono::duration<double>(t1 - t0).count());
    fflush(stdout);

    oroFree((oroDeviceptr)d_light_cache);

    // ---- L1 cache: first 16KB of DAG, copied to separate buffer ----
    constexpr size_t L1_CACHE_BYTES = 16384;
    if (!ls.alloc_l1_cache(L1_CACHE_BYTES, 1 << 20, device_id)) {
        fprintf(stderr, "[KawPow] GPU %d: L1 cache alloc failed\n", device_id);
        ls.free_dag();
        delete kp;
        return false;
    }

    err = oroMemcpy(ls.d_l1_cache, ls.d_dag, L1_CACHE_BYTES, oroMemcpyDeviceToDevice);
    if (err != oroSuccess) {
        fprintf(stderr, "[KawPow] GPU %d: L1 cache copy failed: %s\n",
                device_id, tnn_error_string(err));
        ls.free_dag();
        delete kp;
        return false;
    }

    // Store DAG gen kernel for reuse during epoch rebuilds on worker thread.
    // Modules compiled on init thread work cross-thread (same as mining kernels).
    kp->dag_gen_module = dag_ck.module;
    kp->dag_gen_function = dag_ck.function;

    // Save compilation state for runtime recompilation
    kp->compile_opts = dag_opts;
    kp->compile_device_id = device_id;
    // Compile kernels for multiple periods (variance measurement during autotune)
    {
        auto& opts = dag_opts;
        std::string base_src(hip_kawpow_source::SRC_TNN_HIP_CRYPTO_KAWPOW_KAWPOW_HIP_SOURCE);
        base_src = kawpow_proggen::inject_coin_padding(base_src, *currentKawpowPadding);
        base_src = kawpow_proggen::inject_constant_l1_table(
            base_src,
            reinterpret_cast<const uint32_t*>(ctx->light_cache),
            4096
        );
        base_src = kawpow_proggen::inject_dag_constants(base_src,
            ls.dag_num_items, ls.barrett_m, ls.barrett_shift);

        printf("[KawPow] GPU %d: Compiling %d period kernels for variance bench...\n",
               device_id, KAWPOW_BENCH_PERIODS);
        fflush(stdout);

        auto ct0 = std::chrono::steady_clock::now();
        int compiled_ok = 0;
        for (int p = 0; p < KAWPOW_BENCH_PERIODS; ++p) {
            int block_num = kawpow_bench_block() + p * 3; // consecutive periods from base height
            std::string src = kawpow_proggen::inject_program(base_src, block_num);
            std::string name = "kawpow_p" + std::to_string(p) + ".hip";

            // Compile module — extract mono kernel as primary
            auto ck = compiler.compile_from_source(src, name, "kawpow_hash_kernel", opts, device_id);
            if (ck.function) {
                kp->period_modules[p] = ck.module;
                kp->period_mono_kernels[p] = ck.function;

                // Extract split kernels from the same module
                oroModuleGetFunction(&kp->period_seed_kernels[p],            ck.module, "kawpow_seed_kernel");
                oroModuleGetFunction(&kp->period_progpow_4way_kernels[p],    ck.module, "kawpow_progpow_kernel_2way");
                oroModuleGetFunction(&kp->period_final_kernels[p],           ck.module, "kawpow_final_kernel");

                compiled_ok++;
            } else {
                fprintf(stderr, "[KawPow] GPU %d: Failed to compile period %d\n", device_id, p);
            }
        }
        kp->num_period_kernels = compiled_ok;

        auto ct1 = std::chrono::steady_clock::now();
        printf("[KawPow] GPU %d: %d/%d period kernels compiled in %.1fs\n",
               device_id, compiled_ok, KAWPOW_BENCH_PERIODS,
               std::chrono::duration<double>(ct1 - ct0).count());
        fflush(stdout);

        // Dump first period's fully-injected source for standalone optimization
        {
            int block_num = kawpow_bench_block();
            std::string dump_src = kawpow_proggen::inject_program(base_src, block_num);
            const char* dump_path = "kawpow_bench_kernel.hip";
            FILE* f = fopen(dump_path, "w");
            if (f) {
                fwrite(dump_src.data(), 1, dump_src.size(), f);
                fclose(f);
                printf("[KawPow] GPU %d: Dumped bench kernel source → %s\n", device_id, dump_path);
            }
            fflush(stdout);
        }
    }

    *algo_data = kp;
    return true;
}

// ---------------------------------------------------------------------------
// Helper: timed bench of a single kernel launch pattern (returns MH/s)
// ---------------------------------------------------------------------------
inline double kawpow_timed_bench(
    std::function<void()> launch_fn,
    oroStream_t stream,
    uint32_t batch_size,
    double warmup_s, double bench_s)
{
    // Warmup
    {
        auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warmup_s) {
            launch_fn();
            oroStreamSynchronize(stream);
        }
    }
    // Timed
    uint64_t total = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
        launch_fn();
        oroStreamSynchronize(stream);
        total += batch_size;
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() >= bench_s) break;
    }
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return (double)total / (elapsed * 1e6);
}

// Occupancy-based tune: picks block sizes + strategy (mono vs split)
inline bool kawpow_occupancy_tune(TuningResult& result, const oroDeviceProp_t& props,
                                   int device_id, void* algo_data,
                                   double memory_reserve_mb, double memory_usage_factor)
{
    (void)memory_reserve_mb; (void)memory_usage_factor;
    auto* kp = static_cast<KawPowAlgoData*>(algo_data);
    if (!kp || !kp->period_seed_kernels[0] ||
        !kp->period_progpow_4way_kernels[0] || !kp->period_final_kernels[0]) {
        printf("[KawPow] GPU %d: 2-way kernels not available, cannot tune\n", device_id);
        fflush(stdout);
        return false;
    }

    int CUs = props.multiProcessorCount;

    // Quick occupancy query at default block size to get a starting batch for seed/final sweeps
    int pp_block = 128;
    int pp_bpc = 0;
    oroModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        &pp_bpc, kp->period_progpow_4way_kernels[0], pp_block, kawpow_shared_mem_split(pp_block));
    if (pp_bpc <= 0) { pp_block = 64; pp_bpc = 4; } // safe fallback

    constexpr int BATCH_MULTIPLIER = 64;
    int pp_grid = pp_bpc * CUs * BATCH_MULTIPLIER;
    uint32_t split_batch = (uint32_t)pp_grid * (uint32_t)pp_block / 8u;

    // Allocate intermediate buffer + bench buffers
    if (!kp->ensure_intermediate(split_batch)) {
        printf("[KawPow] GPU %d: Failed to alloc intermediate buffer, using Mono\n", device_id);
        fflush(stdout);
        return true;
    }

    oroStream_t stream = nullptr;
    oroStreamCreate(&stream);
    AlignedDevAlloc bh_alloc{}, bt_alloc{}, bsol_alloc{};

    oroMallocAligned(bh_alloc,   32,             64);
    oroMallocAligned(bt_alloc,   32,             64);
    oroMallocAligned(bsol_alloc, 8 + 1024 * 40,  64);

    uint8_t*  bh   = bh_alloc.aligned;
    uint64_t* bt   = reinterpret_cast<uint64_t*>(bt_alloc.aligned);
    uint64_t* bsol = reinterpret_cast<uint64_t*>(bsol_alloc.aligned);
    
    uint64_t maxt[4] = {~0ULL, ~0ULL, ~0ULL, ~0ULL};
    oroMemcpy(bt, maxt, 32, oroMemcpyHostToDevice);

    uint32_t* hdr = (uint32_t*)bh;
    uint32_t* dag = kp->live[0].d_dag;
    uint32_t* l1 = kp->live[0].d_l1_cache;
    uint64_t nonce = 0;
    uint32_t* rh = nullptr;

    oroFunction_t seed_fn = kp->period_seed_kernels[0];
    oroFunction_t pp_fn   = kp->period_progpow_4way_kernels[0];
    oroFunction_t final_fn = kp->period_final_kernels[0];

    // ---- Sweep seed_tpb ----
    // Capped to 128 to match __launch_bounds__(128, 8) on seed/final kernels
    static constexpr int bookend_tpb_candidates[] = { 32, 64, 96, 128, 192, 256 };
    int best_seed_tpb = 128;
    {
        printf("[KawPow] GPU %d: Seed TPB sweep: ", device_id);
        fflush(stdout);
        double best_ms = 1e9;
        for (int tpb : bookend_tpb_candidates) {
            int sgrid = ((int)split_batch + tpb - 1) / tpb;
            void* args[] = { &hdr, &nonce, &split_batch, &kp->d_intermediate };

            // Quick timing: 20 launches
            oroStreamSynchronize(stream);
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 20; r++)
                oroModuleLaunchKernel(seed_fn, sgrid, 1, 1, tpb, 1, 1, 0, stream, args, nullptr);
            oroStreamSynchronize(stream);
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count() / 20.0;

            printf("%d=%.2fms ", tpb, ms);
            fflush(stdout);
            if (ms < best_ms) { best_ms = ms; best_seed_tpb = tpb; }
        }
        printf("→ best=%d\n", best_seed_tpb);
        fflush(stdout);
    }

    // ---- Sweep final_tpb ----
    int best_final_tpb = 128;
    {
        // First run seed to fill intermediate buffer with valid data
        int sgrid = ((int)split_batch + best_seed_tpb - 1) / best_seed_tpb;
        void* sargs[] = { &hdr, &nonce, &split_batch, &kp->d_intermediate };
        oroModuleLaunchKernel(seed_fn, sgrid, 1, 1, best_seed_tpb, 1, 1, 0, stream, sargs, nullptr);
        // Run progpow to fill digest in intermediate
        void* pargs[] = { &dag, &split_batch, &l1, &kp->d_intermediate };
        oroModuleLaunchKernel(pp_fn, pp_grid, 1, 1, pp_block, 1, 1,
                              kawpow_shared_mem_split(pp_block), stream, pargs, nullptr);
        oroStreamSynchronize(stream);

        printf("[KawPow] GPU %d: Final TPB sweep: ", device_id);
        fflush(stdout);
        double best_ms = 1e9;
        for (int tpb : bookend_tpb_candidates) {
            int fgrid = ((int)split_batch + tpb - 1) / tpb;
            void* args[] = { &nonce, &split_batch, &bt, &bsol, &rh, &kp->d_intermediate };

            oroStreamSynchronize(stream);
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 20; r++)
                oroModuleLaunchKernel(final_fn, fgrid, 1, 1, tpb, 1, 1, 0, stream, args, nullptr);
            oroStreamSynchronize(stream);
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count() / 20.0;

            printf("%d=%.2fms ", tpb, ms);
            fflush(stdout);
            if (ms < best_ms) { best_ms = ms; best_final_tpb = tpb; }
        }
        printf("→ best=%d\n", best_final_tpb);
        fflush(stdout);
    }

    // ---- Build TPB candidate list (occupancy + quick ns/h estimate) ----
    oroFunction_t scan_fn = pp_fn;  // plain 2-way

    constexpr size_t MAX_2D_TPB = 1024;

    struct TpbCandidate { int tpb; int bpc; double ns; };
    TpbCandidate tpb_candidates[32];
    int n_tpb_candidates = 0;
    {
        printf("[KawPow] GPU %d: TPB quick scan (2Way): ", device_id);
        fflush(stdout);

        for (int tbs = 32; tbs <= MAX_2D_TPB; tbs += 32) {  // capped to __launch_bounds__(256, 2)
            int max_blocks = 0;
            oroModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks, scan_fn, tbs, kawpow_shared_mem_split(tbs));
            if (max_blocks <= 0) continue;

            int trial_grid = max_blocks * CUs * 4;
            // 2-way: each 16-thread group processes 2 hashes → /8u
            uint32_t trial_batch = (uint32_t)trial_grid * (uint32_t)tbs / 8u;
            if (!kp->ensure_intermediate(trial_batch)) continue;

            // Seed + 3 warmup + 10 timed
            int sg = ((int)trial_batch + best_seed_tpb - 1) / best_seed_tpb;
            void* sa[] = { &hdr, &nonce, &trial_batch, &kp->d_intermediate };
            oroModuleLaunchKernel(seed_fn, sg, 1, 1, best_seed_tpb, 1, 1, 0, stream, sa, nullptr);
            oroStreamSynchronize(stream);

            void* pargs[] = { &dag, &trial_batch, &l1, &kp->d_intermediate };
            for (int w = 0; w < 3; w++) {
                oroModuleLaunchKernel(scan_fn, trial_grid, 1, 1, tbs, 1, 1,
                                      kawpow_shared_mem_split(tbs), stream, pargs, nullptr);
            }
            oroStreamSynchronize(stream);

            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 10; r++)
                oroModuleLaunchKernel(scan_fn, trial_grid, 1, 1, tbs, 1, 1,
                                      kawpow_shared_mem_split(tbs), stream, pargs, nullptr);
            oroStreamSynchronize(stream);
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count() / 10.0;
            double ns_per_hash = ms * 1e6 / trial_batch;

            printf("%d(%dbpc)=%.1fns/h ", tbs, max_blocks, ns_per_hash);
            fflush(stdout);
            tpb_candidates[n_tpb_candidates++] = {tbs, max_blocks, ns_per_hash};
        }
        printf("\n");
        fflush(stdout);
    }

    // ---- 2D TPB × multiplier sweep helper ----
    static constexpr int mult_candidates[] = {8, 16, 32, 64, 256, 512, 1024, 4096, 8192, 16384, 32768, 65536};
    static constexpr int n_mults = sizeof(mult_candidates) / sizeof(mult_candidates[0]);
    constexpr double MAX_LAUNCH_SEC = 3.0;

    struct SweepWinner { int tpb; int bpc; int mult; double mhs; };

    auto run_2d_sweep = [&](oroFunction_t sweep_fn, const char* label) -> SweepWinner {
        constexpr uint32_t threads_per_hash = 8u;  // 2-way: 16 threads per group, 2 hashes per group
        // Pre-compute max intermediate size needed (use uint64 to avoid overflow)
        size_t max_inter_bytes = 0;
        for (int c = 0; c < n_tpb_candidates; c++) {
            for (int mult : mult_candidates) {
                int grid = tpb_candidates[c].bpc * CUs * mult;
                uint64_t total_threads = (uint64_t)grid * (uint64_t)tpb_candidates[c].tpb;
                if (total_threads / threads_per_hash > UINT32_MAX) continue; // would overflow batch
                uint32_t batch = (uint32_t)(total_threads / threads_per_hash);
                double est = (double)batch * tpb_candidates[c].ns * 1e-9;
                if (est > MAX_LAUNCH_SEC) continue;
                size_t bytes = (size_t)batch * 16 * sizeof(uint32_t);
                if (bytes > max_inter_bytes) max_inter_bytes = bytes;
            }
        }
        kp->ensure_intermediate(max_inter_bytes / (16 * sizeof(uint32_t)));

        printf("[KawPow] GPU %d: 2D TPB x mult sweep — %s:\n", device_id, label);
        printf("  %6s", "mult");
        for (int c = 0; c < n_tpb_candidates; c++)
            printf(" %5d", tpb_candidates[c].tpb);
        printf("  | best\n");
        fflush(stdout);

        SweepWinner best = {tpb_candidates[0].tpb, tpb_candidates[0].bpc, 64, 0};

        for (int mult : mult_candidates) {
            printf("  %5dx", mult);
            fflush(stdout);

            double row_best_mhs = 0;
            int row_best_tpb = 0;

            for (int c = 0; c < n_tpb_candidates; c++) {
                int tbs = tpb_candidates[c].tpb;
                int bpc = tpb_candidates[c].bpc;
                int trial_grid = bpc * CUs * mult;
                uint64_t total_threads = (uint64_t)trial_grid * (uint64_t)tbs;
                if (total_threads / threads_per_hash > UINT32_MAX) {
                    printf("     -"); fflush(stdout); continue;
                }
                uint32_t trial_batch = (uint32_t)(total_threads / threads_per_hash);
                size_t inter_bytes = (size_t)trial_batch * 16 * sizeof(uint32_t);

                double est_sec = (double)trial_batch * tpb_candidates[c].ns * 1e-9;
                if (est_sec > MAX_LAUNCH_SEC || inter_bytes > max_inter_bytes) {
                    printf("     -"); fflush(stdout); continue;
                }

                auto trial_launch = [&]() {
                    kawpow_launch_split(
                        seed_fn, sweep_fn, final_fn,
                        hdr, dag, nonce, trial_batch,
                        (uint64_t*)bt, bsol, rh, l1,
                        kp->d_intermediate,
                        trial_grid, tbs, best_seed_tpb, best_final_tpb,
                        stream);
                };

                // 2 warmups
                trial_launch();
                trial_launch();
                oroStreamSynchronize(stream);

                // 2 timed iters via GPU events
                oroEvent_t ev0 = nullptr, ev1 = nullptr;
                oroEventCreate(&ev0); oroEventCreate(&ev1);
                oroEventRecord(ev0, stream);
                trial_launch();
                oroEventRecord(ev1, stream);
                oroStreamSynchronize(stream);

                float ms = 0;
                oroEventElapsedTime(&ms, ev0, ev1);
                oroEventDestroy(ev0); oroEventDestroy(ev1);

                double mhs = (double)(trial_batch) / ((double)ms * 1e3);
                bool outlier = (best.mhs > 0 && mhs > best.mhs * 10.0);
                printf(" %5.1f", mhs);
                fflush(stdout);

                if (!outlier) {
                    if (mhs > row_best_mhs) { row_best_mhs = mhs; row_best_tpb = tbs; }
                    if (mhs > best.mhs) { best = {tbs, bpc, mult, mhs}; }
                }
            }
            if (row_best_tpb > 0)
                printf("  | %d=%.1f\n", row_best_tpb, row_best_mhs);
            else
                printf("  | -\n");
            fflush(stdout);
        }

        printf("  => %s 2D winner: TPB=%d, mult=%dx (%.2f MH/s)\n\n", label, best.tpb, best.mult, best.mhs);
        fflush(stdout);
        return best;
    };

    // ---- Run 2D sweep for plain 2-way ----
    SweepWinner w = run_2d_sweep(pp_fn, "2Way");

    // ---- Apply winner ----
    result.tune_keys["seed_tpb"] = best_seed_tpb;
    result.tune_keys["final_tpb"] = best_final_tpb;
    result.tune_keys["strategy"] = (int64_t)KawPowStrategy::Split2Way;
    kp->best_strategy = KawPowStrategy::Split2Way;

    pp_block = w.tpb;
    pp_bpc = w.bpc;

    // 2-way: each 16-thread group processes 2 hashes → /8u
    int best_mult = w.mult;
    int final_grid = pp_bpc * CUs * best_mult;
    uint32_t final_batch = (uint32_t)final_grid * (uint32_t)pp_block / 8u;
    result.block_size = pp_block;
    result.num_blocks = final_grid;
    result.batch_size = final_batch;
    result.tune_keys["batch_mult"] = best_mult;
    kp->ensure_intermediate(final_batch);

    printf("[KawPow] GPU %d: Final config: Split2Way, block=%d, %dx mult, grid=%d, batch=%u (%.2f MH/s)\n",
           device_id, pp_block, best_mult, final_grid, final_batch, w.mhs);
    fflush(stdout);

    oroFree((oroDeviceptr)bh);
    oroFree((oroDeviceptr)bt);
    oroFree((oroDeviceptr)bsol);
    oroStreamDestroy(stream);

    return true;
}

// Cleanup: free DAG + kernels for both live states
inline void kawpow_algo_data_cleanup(void* algo_data)
{
    auto* kp = static_cast<KawPowAlgoData*>(algo_data);
    if (kp) {
        oroDeviceSynchronize();

        for (int m = 0; m < 2; ++m) {
            kp->live[m].free_dag();
            kp->live[m].free_module();
        }

        kp->free_intermediate();

        for (int i = 0; i < kp->num_period_kernels; ++i) {
            if (kp->period_modules[i])
                oroModuleUnload(kp->period_modules[i]);
        }

        if (kp->dag_gen_module)
            oroModuleUnload(kp->dag_gen_module);

        delete kp;
    }
}

// ---------------------------------------------------------------------------
// DAG rebuild helper — regenerates DAG + L1 cache + Barrett constants
// for a new epoch. Must be called on GPU thread with context bound.
// ---------------------------------------------------------------------------
inline bool kawpow_rebuild_dag(KawPowLiveState& ls, KawPowAlgoData* kp, int new_epoch, int device_id, bool is_dev)
{
    const char* tag = is_dev ? "DEV " : "";

    oroDeviceSynchronize();

    printf("[KawPow] GPU %d: %sEpoch change %d -> %d, rebuilding DAG...\n",
           device_id, tag, ls.current_epoch, new_epoch);
    fflush(stdout);

    auto ctx = ethash::create_epoch_context(new_epoch);
    if (!ctx) {
        fprintf(stderr, "[KawPow] GPU %d: Failed to create epoch context for epoch %d\n",
                device_id, new_epoch);
        return false;
    }

    // Free old DAG + L1 before reallocating
    ls.free_dag();

    // Recalculate sizes + Barrett constants
    ls.dag_num_items = (uint32_t)(ctx->full_dataset_num_items / 2);
    size_t dag_words = (size_t)ls.dag_num_items * 64;
    ls.dag_size_bytes = dag_words * sizeof(uint32_t);
    {
        uint32_t d = ls.dag_num_items;
        ls.barrett_m = (uint32_t)((1ULL << 32) / d);
        ls.barrett_shift = 0;
    }

    {
        size_t free_mem, total_mem;
        oroMemGetInfo(&free_mem, &total_mem);
        printf("[KawPow] GPU %d: %sDAG epoch %d: %u items (%.1f GB), VRAM: %.0f/%.0f MB free\n",
               device_id, tag, new_epoch, ls.dag_num_items,
               ls.dag_size_bytes / (1024.0 * 1024.0 * 1024.0),
               free_mem / (1024.0 * 1024.0), total_mem / (1024.0 * 1024.0));
        fflush(stdout);
    }
    if (!ls.alloc_dag(ls.dag_size_bytes, 1 << 20, device_id)) {
        fprintf(stderr, "[KawPow] GPU %d: %sDAG alloc failed (%.1f GB)\n",
                device_id, tag, ls.dag_size_bytes / (1024.0 * 1024.0 * 1024.0));
        return false;
    }

    // Upload light cache
    uint32_t num_cache_items = (uint32_t)ctx->light_cache_num_items;
    size_t cache_bytes = (size_t)num_cache_items * 64;
    uint32_t* d_light_cache = nullptr;
    auto err = oroMalloc((oroDeviceptr*)&d_light_cache, cache_bytes);
    if (err != oroSuccess) {
        fprintf(stderr, "[KawPow] GPU %d: %sLight cache alloc failed: %s\n", device_id, tag, tnn_error_string(err));
        ls.free_dag();
        return false;
    }
    oroMemcpyHtoD((oroDeviceptr)d_light_cache, ctx->light_cache, cache_bytes);

    printf("[KawPow] GPU %d: %sLight cache uploaded (%.1f MB, %u items)\n",
           device_id, tag, cache_bytes / (1024.0 * 1024.0), num_cache_items);
    fflush(stdout);

    // Launch DAG generation kernel
    oroFunction_t dag_fn = kp->dag_gen_function;
    if (!dag_fn) {
        fprintf(stderr, "[KawPow] GPU %d: DAG gen kernel not available\n", device_id);
        oroFree((oroDeviceptr)d_light_cache);
        ls.free_dag();
        return false;
    }

    (void)oroGetLastError();

    uint32_t dag_items = ls.dag_num_items;
    int block_size = 256;
    int grid_size = (dag_items + block_size - 1) / block_size;
    void* dag_args[] = { &d_light_cache, &ls.d_dag, &num_cache_items, &dag_items };

    auto t0 = std::chrono::steady_clock::now();
    err = oroModuleLaunchKernel(dag_fn, grid_size, 1, 1, block_size, 1, 1,
                                 0, nullptr, dag_args, nullptr);
    if (err != oroSuccess) {
        fprintf(stderr, "[KawPow] GPU %d: DAG gen launch failed: %s\n",
                device_id, tnn_error_string(err));
        oroFree((oroDeviceptr)d_light_cache);
        ls.free_dag();
        return false;
    }
    oroError_t sync_err = oroDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double gen_secs = std::chrono::duration<double>(t1 - t0).count();

    if (sync_err != oroSuccess) {
        fprintf(stderr, "[KawPow] GPU %d: %sDAG gen kernel error: %s\n",
                device_id, tag, tnn_error_string(sync_err));
        oroFree((oroDeviceptr)d_light_cache);
        ls.free_dag();
        return false;
    }

    printf("[KawPow] GPU %d: %sDAG generated in %.2fs\n", device_id, tag, gen_secs);
    fflush(stdout);

    oroFree((oroDeviceptr)d_light_cache);

    // L1 cache: first 16KB of DAG
    err = oroMalloc((oroDeviceptr*)&ls.d_l1_cache, 16384);
    if (err != oroSuccess) {
        fprintf(stderr, "[KawPow] GPU %d: L1 cache alloc failed: %s\n", device_id, tnn_error_string(err));
        ls.free_dag();
        return false;
    }
    oroMemcpyDtoD((oroDeviceptr)ls.d_l1_cache, (oroDeviceptr)ls.d_dag, 16384);

    ls.current_epoch = new_epoch;
    ls.current_period = -1;  // force period recompile (Barrett constants changed)
    return true;
}

// ---------------------------------------------------------------------------
// Deps-changed hook: rebuild DAG on epoch change, recompile kernel on period change.
// Dep key: "period" (= block_number / 3).  Epoch derived as period / 2500.
// ---------------------------------------------------------------------------
inline bool kawpow_deps_changed(
    const std::unordered_map<std::string, int64_t>& new_deps,
    const std::unordered_map<std::string, int64_t>& old_deps,
    KernelMap& kernels,
    void* algo_data,
    int device_id,
    bool is_dev)
{
    (void)old_deps;
    auto* kp = static_cast<KawPowAlgoData*>(algo_data);
    if (!kp) return false;

    // Single shared live state — session-based dev fee means only one mode
    // is active at a time, so DAG + kernels share live[0].
    auto& ls = kp->live[0];
    const char* tag = is_dev ? "DEV " : "";

    auto it = new_deps.find("period");
    if (it == new_deps.end()) return true;

    int new_period = (int)it->second;
    int new_epoch = (new_period * 3) / ETHASH_EPOCH_LENGTH;

    // Epoch change → defer to main thread via hook system
    if (new_epoch != ls.current_epoch) {
        kp->pending_rebuild_epoch = new_epoch;
        kp->pending_rebuild_is_dev = is_dev;
        printf("[KawPow] GPU %d: %sEpoch change %d -> %d — requesting context rebuild\n",
               device_id, is_dev ? "DEV " : "", ls.current_epoch, new_epoch);
        fflush(stdout);
        return false;
    }

    // Period change → recompile kernel
    if (new_period == ls.current_period) return true;

    int canonical_block = new_period * 3;

    printf("[KawPow] GPU %d: %sPeriod change %d → %d, recompiling...\n",
           device_id, tag, ls.current_period, new_period);
    fflush(stdout);

    auto& compiler = RTCCompiler::instance();

    std::string base_src(hip_kawpow_source::SRC_TNN_HIP_CRYPTO_KAWPOW_KAWPOW_HIP_SOURCE);
    base_src = kawpow_proggen::inject_coin_padding(base_src, *currentKawpowPadding);
    base_src = kawpow_proggen::inject_dag_constants(base_src, ls.dag_num_items,
                                                     ls.barrett_m, ls.barrett_shift);
    std::string src = kawpow_proggen::inject_program(base_src, canonical_block);
    std::string name = "kawpow_live_" + std::string(is_dev ? "dev_" : "") + "p" + std::to_string(new_period) + ".hip";

    auto ck = compiler.compile_from_source(src, name, "kawpow_hash_kernel",
                                            kp->compile_opts, kp->compile_device_id);
    if (!ck.function) {
        fprintf(stderr, "[KawPow] GPU %d: %sFailed to compile period %d\n", device_id, tag, new_period);
        return false;
    }

    // Unload old live module
    ls.free_module();

    ls.live_module = ck.module;
    ls.live_mono_kernel = ck.function;
    oroModuleGetFunction(&ls.live_seed_kernel,          ck.module, "kawpow_seed_kernel");
    oroModuleGetFunction(&ls.live_progpow_2way_kernel, ck.module, "kawpow_progpow_kernel_2way");
    oroModuleGetFunction(&ls.live_final_kernel,        ck.module, "kawpow_final_kernel");

    // Update the kernel map so execute_fn uses the new kernels
    kernels["kawpow_hash_kernel"] = ls.live_mono_kernel;
    if (ls.live_seed_kernel)          kernels["kawpow_seed_kernel"]          = ls.live_seed_kernel;
    if (ls.live_progpow_2way_kernel)  kernels["kawpow_progpow_kernel_2way"] = ls.live_progpow_2way_kernel;
    if (ls.live_final_kernel)         kernels["kawpow_final_kernel"]         = ls.live_final_kernel;

    printf("[KawPow] GPU %d: %sRecompiled for period %d\n", device_id, tag, new_period);
    fflush(stdout);

    ls.current_period = new_period;
    return true;
}

// Execute: dispatch based on strategy (Mono or Split)
inline bool kawpow_execute(
    const KernelMap& kernels,
    const KernelLaunchContext& ctx)
{
    auto* kp = static_cast<KawPowAlgoData*>(ctx.algo_data);
    if (!kp) return false;

    auto& ls = kp->live[0];
    if (!ls.d_dag) return false;

    auto strategy = static_cast<KawPowStrategy>(ctx.get_tune_key("strategy", (int64_t)KawPowStrategy::Split2Way));

    uint32_t* header = (uint32_t*)ctx.d_input;
    uint32_t* dag = ls.d_dag;
    uint64_t nonce_start = ctx.nonce_start;
    uint32_t batch_size = ctx.batch_size;
    uint64_t* target = ctx.d_difficulty_target;
    uint64_t* solutions = ctx.d_solutions;
    uint32_t* result_hashes = nullptr;
    uint32_t* l1_cache = ls.d_l1_cache;

    switch (strategy) {
    case KawPowStrategy::Split2Way: {
        auto it_seed = kernels.find("kawpow_seed_kernel");
        auto it_pp   = kernels.find("kawpow_progpow_kernel_2way");
        auto it_fin  = kernels.find("kawpow_final_kernel");
        if (it_seed == kernels.end() || it_pp == kernels.end() || it_fin == kernels.end()) {
            fprintf(stderr, "[KawPow] GPU %d: Split2Way kernels not found in map (seed=%d pp=%d fin=%d)\n",
                    kp->compile_device_id, it_seed != kernels.end(), it_pp != kernels.end(), it_fin != kernels.end());
            return false;
        }

        if (!kp->ensure_intermediate(batch_size)) {
            fprintf(stderr, "[KawPow] GPU %d: ensure_intermediate(%u) failed\n", kp->compile_device_id, batch_size);
            return false;
        }

        int seed_tpb  = (int)ctx.get_tune_key("seed_tpb", 128);
        int final_tpb = (int)ctx.get_tune_key("final_tpb", 128);
        int seed_grid = ((int)batch_size + seed_tpb - 1) / seed_tpb;
        int final_grid = ((int)batch_size + final_tpb - 1) / final_tpb;
        size_t smem = kawpow_shared_mem_split(ctx.block_size);

        kawpow_launch_split(
            it_seed->second, it_pp->second, it_fin->second,
            header, dag, nonce_start, batch_size,
            target, solutions, result_hashes, l1_cache,
            kp->d_intermediate,
            ctx.num_blocks, ctx.block_size,
            seed_tpb, final_tpb,
            ctx.stream);
        return true;
    }

    case KawPowStrategy::Mono:
    default: {
        auto it = kernels.find("kawpow_hash_kernel");
        if (it == kernels.end()) return false;

        uint32_t total_threads = batch_size * 16u;
        uint32_t grid = (total_threads + ctx.block_size - 1) / ctx.block_size;
        uint32_t block_number = kp->block_number;

        void* args[] = {
            &header, &dag, &nonce_start, &batch_size,
            &target, &solutions, &block_number,
            &result_hashes, &l1_cache,
        };

        auto err = oroModuleLaunchKernel(
            it->second, grid, 1, 1,
            ctx.block_size, 1, 1,
            kawpow_shared_mem_mono(ctx.block_size), ctx.stream, args, nullptr);
        return (err == oroSuccess);
    }
    }
}

// ---------------------------------------------------------------------------
// Post-tune: measure per-period hashrate variance at the winning config
// ---------------------------------------------------------------------------
inline void kawpow_post_tune(const TuningResult& result,
                              const oroDeviceProp_t& props,
                              int device_id, void* algo_data,
                              int warmup_count, int timed_count)
{
    (void)props; (void)warmup_count; (void)timed_count;
    auto* kp = static_cast<KawPowAlgoData*>(algo_data);
    if (!kp || kp->num_period_kernels == 0) return;

    constexpr double BENCH_DURATION_S = 30.0;

    printf("\n[KawPow] GPU %d: Sustained bench (%.0fs, %d programs, block=%d, batch=%u)\n",
           device_id, BENCH_DURATION_S, kp->num_period_kernels,
           result.block_size, result.batch_size);
    fflush(stdout);

    // Allocate minimal test buffers
    uint8_t* d_header = nullptr;
    uint64_t* d_target = nullptr;
    uint64_t* d_solutions = nullptr;
    oroStream_t stream = nullptr;

    oroStreamCreate(&stream);
    oroMalloc((oroDeviceptr*)&d_header, 32);
    oroMalloc((oroDeviceptr*)&d_target, 32);
    oroMalloc((oroDeviceptr*)&d_solutions, 8 + 1024 * 40);

    oroMemset((oroDeviceptr)d_header, 0, 32);
    uint64_t max_tgt[4] = {~0ULL, ~0ULL, ~0ULL, ~0ULL};
    oroMemcpy(d_target, max_tgt, 32, oroMemcpyHostToDevice);

    uint32_t batch_size = result.batch_size;
    uint32_t grid = result.num_blocks;  // grid from tune (already accounts for 2-way)
    size_t shared_mem = kawpow_shared_mem(result.block_size);
    int num_kernels = kp->num_period_kernels;

    // Build args — nonce_start and block_number are updated in the loop
    uint32_t* header = (uint32_t*)d_header;
    uint32_t* dag = kp->live[0].d_dag;
    uint64_t nonce_start = 0;
    uint32_t block_number = kawpow_bench_block();
    uint32_t* result_hashes = nullptr;
    uint32_t* l1_cache = kp->live[0].d_l1_cache;

    void* args[] = {
        &header, &dag, &nonce_start, &batch_size,
        &d_target, &d_solutions, &block_number,
        &result_hashes, &l1_cache,
    };

    // ---- Sustained bench with winning strategy ----
    int seed_tpb = result.tune_keys.count("seed_tpb") ? (int)result.tune_keys.at("seed_tpb") : 128;
    int final_tpb = result.tune_keys.count("final_tpb") ? (int)result.tune_keys.at("final_tpb") : 128;

    printf("  Strategy: Split2Way\n");
    fflush(stdout);

    // Ensure intermediate buffer for split pipeline
    kp->ensure_intermediate(batch_size);

    // Build launch lambda — cycles through compiled period kernels
    int period_idx = 0;
    auto launch = [&]() {
        int p = period_idx % num_kernels;
        period_idx++;
        block_number = kawpow_bench_block() + p * 3;

        oroFunction_t pp_kern = kp->period_progpow_4way_kernels[p];
        kawpow_launch_split(
            kp->period_seed_kernels[p], pp_kern, kp->period_final_kernels[p],
            header, dag, nonce_start, batch_size,
            d_target, d_solutions, result_hashes, l1_cache,
            kp->d_intermediate,  // shared intermediate buffer (bench only)
            grid, result.block_size, seed_tpb, final_tpb,
            stream);
    };

    // Warmup: 3 launches
    for (int w = 0; w < 3; ++w) {
        oroMemsetAsync((oroDeviceptr)d_solutions, 0, 4, stream);
        launch();
    }
    oroStreamSynchronize(stream);
    auto t_start = std::chrono::steady_clock::now();
    auto t_last_print = t_start;
    uint64_t total_hashes = 0;
    uint64_t interval_hashes = 0;
    double peak_mhs = 0;

    printf("  [time]  instant    avg      peak\n");
    fflush(stdout);

    while (true) {
        oroMemsetAsync((oroDeviceptr)d_solutions, 0, 4, stream);
        nonce_start = total_hashes;
        launch();
        oroStreamSynchronize(stream);

        total_hashes += batch_size;
        interval_hashes += batch_size;

        auto now = std::chrono::steady_clock::now();
        double elapsed_total = std::chrono::duration<double>(now - t_start).count();
        double elapsed_interval = std::chrono::duration<double>(now - t_last_print).count();

        if (elapsed_interval >= 1.0) {
            double instant_mhs = (double)interval_hashes / (elapsed_interval * 1e6);
            double avg_mhs = (double)total_hashes / (elapsed_total * 1e6);
            if (instant_mhs > peak_mhs) peak_mhs = instant_mhs;

            printf("  [%4.0fs]  %6.2f    %6.2f    %6.2f MH/s\n",
                   elapsed_total, instant_mhs, avg_mhs, peak_mhs);
            fflush(stdout);

            interval_hashes = 0;
            t_last_print = now;
        }

        if (elapsed_total >= BENCH_DURATION_S) break;
    }

    double total_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    double final_avg = (double)total_hashes / (total_s * 1e6);

    printf("  ----------------------------------------\n");
    printf("  %llu hashes in %.1fs — avg: %.2f MH/s, peak: %.2f MH/s\n",
           (unsigned long long)total_hashes, total_s, final_avg, peak_mhs);
    fflush(stdout);

    oroFree((oroDeviceptr)d_header);
    oroFree((oroDeviceptr)d_target);
    oroFree((oroDeviceptr)d_solutions);
    oroStreamDestroy(stream);

    // Free bench-period modules — no longer needed after tuning
    for (int i = 0; i < kp->num_period_kernels; ++i) {
        if (kp->period_modules[i]) {
            oroModuleUnload(kp->period_modules[i]);
            kp->period_modules[i] = nullptr;
        }
        kp->period_mono_kernels[i] = nullptr;
        kp->period_seed_kernels[i] = nullptr;
        kp->period_progpow_2way_kernels[i] = nullptr;
        kp->period_progpow_4way_kernels[i] = nullptr;
        kp->period_final_kernels[i] = nullptr;
    }
    kp->num_period_kernels = 0;

    printf("[KawPow] GPU %d: Freed bench-period modules\n", device_id);
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// Context rebuild predicates — used by mine_batch to detect + handle pending
// epoch changes that require a fresh GPU context for large allocations.
// ---------------------------------------------------------------------------
inline bool kawpow_needs_context_rebuild(const void* algo_data)
{
    auto* kp = static_cast<const KawPowAlgoData*>(algo_data);
    return kp && kp->pending_rebuild_epoch >= 0;
}

inline bool kawpow_context_rebuild(
    KernelMap& kernels,
    void*& algo_data,
    int device_id,
    bool is_dev)
{
    auto* kp = static_cast<KawPowAlgoData*>(algo_data);
    if (!kp || kp->pending_rebuild_epoch < 0) return true;

    int new_epoch = kp->pending_rebuild_epoch;
    bool rebuild_is_dev = kp->pending_rebuild_is_dev;
    kp->pending_rebuild_epoch = -1;  // consume

    // Reuse the original DAG gen function from pre_tune — no module reload.
    // Same thread, same context, original module still loaded.
    auto& ls = kp->live[0];
    if (!kawpow_rebuild_dag(ls, kp, new_epoch, device_id, rebuild_is_dev))
        return false;

    return true;
}

inline AlgoConfig KAWPOW_CONFIG = {
    .name = "kawpow",
    .source_path = "src/tnn_hip/crypto/kawpow/kawpow.hip",
    .source = hip_kawpow_source::SRC_TNN_HIP_CRYPTO_KAWPOW_KAWPOW_HIP_SOURCE.data(),

    .kernel_names = {"kawpow_hash_kernel", "kawpow_seed_kernel", "kawpow_progpow_kernel_2way", "kawpow_progpow_kernel_4way", "kawpow_final_kernel"},
    .kernel_name = "",

    .rtc_headers = build_rtc_headers(hip_embedded::KAWPOW_HEADERS, hip_embedded::COMMON_HEADERS),
    .template_size = 32,
    .hash_size = 32,
    .nonce_size = 8,
    .scratch_per_hash = 16 * sizeof(uint32_t), // intermediate buffer: 16 uint32 per hash
    .preferred_block_size = 128,
    .algo_id = ALGO_KAWPOW,
    .calc_shared_mem = kawpow_shared_mem,

    .category = AlgoCategory::MemoryHard,
    .enable_reg_tuning = false,

    .amd_blocks = {32, 1024, 32},
    .nvidia_blocks = {32, 1024, 32},
    .target_batch_time_ms = 200.0,
    .max_batch_time_ms = 500.0,
    .min_batch_time_ms = 0.1,
    .enable_autotune = true,
    .autotune_warmup = 2,
    .autotune_iterations = 3,
    .batch_step_denom = 2,
    .memory_reserve_mb = 256.0,
    .memory_usage_factor = 1.0,

    .execute_fn = kawpow_execute,

    .strategy_variants = {},
    .strategy_names = {},
    .strategy_bottleneck_kernels = {},

    .occupancy_threshold = 0.7,

    // KawPow-specific hooks
    .source_transform_fn = kawpow_source_transform,
    .pre_tune_fn = kawpow_pre_tune,
    .occupancy_tune_fn = kawpow_occupancy_tune,
    .post_tune_fn = kawpow_post_tune,
    .algo_data_cleanup_fn = kawpow_algo_data_cleanup,
    .deps_changed_fn = kawpow_deps_changed,
    .main_thread_fn = kawpow_context_rebuild,
    .needs_main_thread_fn = kawpow_needs_context_rebuild,
    .dev_fee_session_based = true,
};
#endif // TNN_KAWPOW

// ============================================================================
// Algorithm Registry
// ============================================================================
class AlgoRegistry
{
public:
  static AlgoRegistry &instance()
  {
    static AlgoRegistry inst;
    return inst;
  }

  std::unique_ptr<IGPUAlgorithm> create(const std::string &name)
  {
    if (name == "xelis_v3")
    {
      return std::make_unique<GPUAlgorithm>(XELIS_V3_CONFIG);
    }
#ifdef TNN_KAWPOW
    if (name == "kawpow")
    {
      return std::make_unique<GPUAlgorithm>(KAWPOW_CONFIG);
    }
#endif
    return nullptr;
  }

  std::vector<std::string> list_algorithms() const
  {
    return {
      "xelis_v3",
#ifdef TNN_KAWPOW
      "kawpow",
#endif
    };
  }
};
