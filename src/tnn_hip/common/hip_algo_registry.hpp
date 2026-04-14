#pragma once
#include "gpu_algo_impl.hpp"
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
#include <algo_definitions.h>
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

    for (int tpb = 32; tpb <= 1024; tpb += 32)
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
          for (int tpb = 32; tpb <= 1024; tpb += 32)
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
          for (int tpb = 32; tpb <= 1024; tpb += 32)
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

    .amd_blocks = {32, 1024, 32},
    .nvidia_blocks = {32, 1024, 32},
    .target_batch_time_ms = 1250.0,
    .max_batch_time_ms = 2000.0,
    .min_batch_time_ms = 100.0,
    .enable_autotune = true,
    .autotune_warmup = 1,
    .autotune_iterations = 1,
    .memory_reserve_mb = 32.0,
    .memory_usage_factor = 1.0,

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
// KawPow Configuration (single strategy)
// ============================================================================
#ifdef TNN_KAWPOW

inline size_t kawpow_shared_mem(int block_size)
{
  return 0;
}

// Single-strategy execute: just launch the monolithic kernel
inline bool kawpow_execute(
    const KernelMap& kernels,
    const KernelLaunchContext& ctx)
{
    auto it = kernels.find("kawpow_hash_kernel");
    if (it == kernels.end()) return false;

    uint32_t grid = (ctx.batch_size + ctx.block_size - 1) / ctx.block_size;

    // TODO: wire up DAG + l1_cache device pointers and block_number/dag_num_items
    // For now this is a stub — the kernel itself is a placeholder
    uint8_t* header = ctx.d_input;
    uint32_t* dag = nullptr;
    uint32_t* l1_cache = nullptr;
    uint64_t nonce_start = ctx.nonce_start;
    uint32_t batch_size = ctx.batch_size;
    uint64_t* target = ctx.d_difficulty_target;
    uint64_t* solutions = ctx.d_solutions;
    int block_number = 0;
    int dag_num_items = 0;

    void* args[] = {
        &header,
        &dag,
        &l1_cache,
        &nonce_start,
        &batch_size,
        &target,
        &solutions,
        &block_number,
        &dag_num_items,
    };

    auto err = oroModuleLaunchKernel(
        it->second, grid, 1, 1,
        ctx.block_size, 1, 1,
        0, ctx.stream, args, nullptr);
    return (err == oroSuccess);
}

inline AlgoConfig KAWPOW_CONFIG = {
    .name = "kawpow",
    .source_path = "src/tnn_hip/crypto/kawpow/kawpow.hip",
    .source = hip_kawpow_source::SRC_TNN_HIP_CRYPTO_KAWPOW_KAWPOW_HIP_SOURCE.data(),

    .kernel_names = {"kawpow_hash_kernel"},
    .kernel_name = "",

    .rtc_headers = build_rtc_headers(hip_embedded::COMMON_HEADERS),
    .template_size = 32,   // header hash (will be refined for actual stratum)
    .hash_size = 32,
    .nonce_size = 8,
    .scratch_per_hash = 0, // DAG is shared, not per-hash scratch
    .preferred_block_size = 128,
    .algo_id = ALGO_KAWPOW,
    .calc_shared_mem = kawpow_shared_mem,

    .category = AlgoCategory::MemoryHard,
    .enable_reg_tuning = false,

    .amd_blocks = {64, 256, 64},
    .nvidia_blocks = {128, 256, 128},
    .target_batch_time_ms = 200.0,
    .max_batch_time_ms = 500.0,
    .min_batch_time_ms = 50.0,
    .enable_autotune = true,
    .autotune_warmup = 2,
    .autotune_iterations = 3,
    .memory_reserve_mb = 256.0,   // DAG can be 2-4 GB
    .memory_usage_factor = 0.85,

    .execute_fn = kawpow_execute,

    // Single strategy — no sweep
    .strategy_variants = {},
    .strategy_names = {},
    .strategy_bottleneck_kernels = {},

    .occupancy_threshold = 0.70,
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
