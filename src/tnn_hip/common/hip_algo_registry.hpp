#pragma once
#include "gpu_algo_impl.hpp"
#include "oro_seh_wrappers.hpp"
#include <memory>
#include <functional>

#include "tnn_hip_common_embedded.hpp"
#ifdef TNN_XELISHASH
#include "xelis_embedded_headers.hpp"
#include "xelis-hash-v3.hip.hpp"
#endif

// ============================================================================
// Xelis V3 Shared Memory Calculator
// ============================================================================
inline size_t xelis_v3_shared_mem(int block_size)
{
  return 256; // LDS S-box for AES rounds in stage_3_hybrid_v2
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
  Sep = 2,      // all 3 separate
  Neo = 3,      // s1 separate, s3+b3 fused
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
  int num_blocks = chunk_batch / stage1_block_size;

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

  int s1_tpb = xelis_stage1_coop_tpb(dev);
  int stage1_block_size = cooperative ? s1_tpb : std::min(ctx.block_size, 32);
  size_t shared_mem = cooperative ? (s1_tpb * 176) : 0;

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
      // Align chunk down to block size (last chunk may be smaller)
      chunk = (chunk / stage1_block_size) * stage1_block_size;
      if (chunk == 0)
        break;

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
    int blake3_block_size = xelis_b3_coop_tpb(dev);
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
  // Only relevant for strategies that use separate stage1
  auto strat = static_cast<XelisStrategy>(result.strategy);
  if (strat != XelisStrategy::Sep && strat != XelisStrategy::Neo)
    return true; // no probe needed

  const char *s1_name = xelis_pick_stage1(device_id);
  auto it = kernels.find(s1_name);
  if (it == kernels.end())
    return false;

  bool cooperative = (std::strcmp(s1_name, "xelis_stage1_cooperative") == 0);
  int s1_tpb = xelis_stage1_coop_tpb(device_id);
  int block_size = cooperative ? s1_tpb : 32;
  size_t shared_mem = cooperative ? (s1_tpb * 176) : 0;

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
    // Too few points to detect a knee
    (void)oro_safe_free((oroDeviceptr)probe_input);
    return true;
  }

  TuneOutputBuffer out(device_id);
  out.printf("[AUTOTUNE] GPU %d: S1 bandwidth sweep (%zu points, %u..%u hashes)\n",
             device_id, probe_batches.size(), probe_batches.front(), probe_batches.back());

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

    int num_blocks = probe_batch / block_size;
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

  return true;
}

inline bool xelis_v3_execute(
    const KernelMap &kernels,
    const KernelLaunchContext &ctx)
{
  int dev = 0;
  (void)oroGetDevice(&dev);

  const auto strategy = static_cast<XelisStrategy>(ctx.strategy);
  uint32_t scratch_offset = 0;

  switch (strategy)
  {

  case XelisStrategy::Mono:
    return default_monolithic_execute(kernels, ctx);

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
        (void *)&scratch_offset};

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
        (void *)&ctx.nonce_start};

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
        (void *)&ctx.nonce_start};

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

  default:
    return default_monolithic_execute(kernels, ctx);
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
        "xelis_blake3_warp_coop_batch"       // Sep blake3 (warp-cooperative: global-CV, no smem)
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
    .max_batch_time_ms = 3125.0,
    .min_batch_time_ms = 100.0,
    .enable_autotune = true,
    .autotune_warmup = 1,
    .autotune_iterations = 1,
    .memory_reserve_mb = 32.0,
    .memory_usage_factor = 1.0,

    .execute_fn = xelis_v3_execute,

    // Sep + Neo — cooperative blake3 makes Mono/Baseline obsolete
    .strategy_variants = {// (uint8_t)XelisStrategy::Mono,
                          // (uint8_t)XelisStrategy::Baseline,
                          (uint8_t)XelisStrategy::Sep, (uint8_t)XelisStrategy::Neo},
    .strategy_names = {/*"Mono", "Baseline",*/ "Sep", "Neo"},

    // Bottleneck kernel per strategy (for occupancy queries)
    .strategy_bottleneck_kernels = {// "xelis_hash_v3_kernel",
                                    // "xelis_s13_noblake_kernel",
                                    "xelis_s3_hybrid_v2_noblake_kernel", "xelis_s3b3_hybrid_v2_kernel"},

    .occupancy_threshold = 1.0,

    .tune_key_probe_fn = xelis_tune_key_probe,
};

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
    return nullptr;
  }

  std::vector<std::string> list_algorithms() const
  {
    return {"xelis_v3"};
  }
};
