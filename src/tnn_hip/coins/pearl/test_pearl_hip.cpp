#include "test_pearl_hip.h"

#include <tnn_hip/crypto/pearl/pearl_pouw_defs.h>
#include <tnn_hip/crypto/pearl/pearl_pouw_vectors.inc>

#include <BLAKE3/c/blake3.h>

#include <algo_definitions.h>
#include <tnn_log.hpp>

#ifdef TNN_HIP
#include <tnn_hip/common/gpu_algo.hpp>
#include <tnn_hip/common/gpu_rtc.hpp>
#include "iris_embedded_headers.hpp"
#include "pearl-noise-dense-test.hip.hpp"
#include "pearl-expand-jackpot.hip.hpp"
#include "pearl_embedded_headers.hpp"
#include "tnn_hip_common_embedded.hpp"
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace tnn::pearl {

namespace {

void append_u16_le(std::vector<uint8_t>& out, uint16_t v)
{
  out.push_back(static_cast<uint8_t>(v & 0xff));
  out.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
}

void append_u32_le(std::vector<uint8_t>& out, uint32_t v)
{
  out.push_back(static_cast<uint8_t>(v & 0xff));
  out.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
  out.push_back(static_cast<uint8_t>((v >> 16) & 0xff));
  out.push_back(static_cast<uint8_t>((v >> 24) & 0xff));
}

template <size_t N>
std::string hex_bytes(const std::array<uint8_t, N>& bytes)
{
  static constexpr char lut[] = "0123456789abcdef";
  std::string out;
  out.resize(N * 2);
  for (size_t i = 0; i < N; ++i)
  {
    out[i * 2] = lut[bytes[i] >> 4];
    out[i * 2 + 1] = lut[bytes[i] & 0x0f];
  }
  return out;
}

bool check_bytes(const char* tag, const char* name, const std::string& got, const char* want)
{
  if (got == want)
  {
    TNN_LOG_INFO("%s %s OK\n", tag, name);
    return true;
  }

  TNN_LOG_ERROR("%s %s mismatch\n", tag, name);
  TNN_LOG_ERROR("%s   got:  %s\n", tag, got.c_str());
  TNN_LOG_ERROR("%s   want: %s\n", tag, want);
  return false;
}

bool check_u32_list(const char* tag, const char* name, const std::vector<uint32_t>& got, const std::vector<uint32_t>& want)
{
  if (got == want)
  {
    TNN_LOG_INFO("%s %s OK\n", tag, name);
    return true;
  }

  TNN_LOG_ERROR("%s %s mismatch\n", tag, name);
  return false;
}

uint8_t hex_nibble(char c)
{
  if (c >= '0' && c <= '9')
    return static_cast<uint8_t>(c - '0');
  if (c >= 'a' && c <= 'f')
    return static_cast<uint8_t>(10 + c - 'a');
  if (c >= 'A' && c <= 'F')
    return static_cast<uint8_t>(10 + c - 'A');
  return 0;
}

Hash256 hash_from_hex(const char* hex)
{
  Hash256 out{};
  for (size_t i = 0; i < out.size(); ++i)
  {
    out[i] = static_cast<uint8_t>((hex_nibble(hex[i * 2]) << 4) | hex_nibble(hex[i * 2 + 1]));
  }
  return out;
}

using I32Matrix = std::vector<std::vector<int32_t>>;

template <size_t RowCount>
I32Matrix matrix_from_hex_rows(const char* const (&rows)[RowCount], size_t cols)
{
  I32Matrix out;
  out.reserve(RowCount);

  for (const char* row_hex : rows)
  {
    std::vector<int32_t> row;
    row.reserve(cols);
    for (size_t i = 0; i < cols; ++i)
    {
      const uint8_t raw = static_cast<uint8_t>((hex_nibble(row_hex[i * 2]) << 4) | hex_nibble(row_hex[i * 2 + 1]));
      row.push_back(raw < 128 ? static_cast<int32_t>(raw) : static_cast<int32_t>(raw) - 256);
    }
    out.push_back(std::move(row));
  }

  return out;
}

std::array<uint32_t, 16> compute_jackpot_words(
  const I32Matrix& s_a,
  const I32Matrix& s_b,
  const I32Matrix& noise_a,
  const I32Matrix& noise_b,
  size_t k,
  size_t rank)
{
  constexpr size_t jackpot_size = 16;
  constexpr uint32_t lrot_per_tile = 13;

  const size_t h = s_a.size();
  const size_t w = s_b.size();
  std::vector<std::vector<int32_t>> jackpot(h, std::vector<int32_t>(w, 0));
  std::array<uint32_t, jackpot_size> jackpot_msg{};

  for (size_t ll = rank; ll <= k; ll += rank)
  {
    for (size_t u = 0; u < h; ++u)
    {
      for (size_t v = 0; v < w; ++v)
      {
        for (size_t l = ll - rank; l < ll; ++l)
        {
          jackpot[u][v] += (s_a[u][l] + noise_a[u][l]) * (s_b[v][l] + noise_b[v][l]);
        }
      }
    }

    uint32_t xored_tile = 0;
    for (const auto& row : jackpot)
    {
      for (int32_t value : row)
      {
        xored_tile ^= static_cast<uint32_t>(value);
      }
    }

    const size_t tid = (ll / rank - 1) % jackpot_size;
    jackpot_msg[tid] = (jackpot_msg[tid] << lrot_per_tile) | (jackpot_msg[tid] >> (32 - lrot_per_tile));
    jackpot_msg[tid] ^= xored_tile;
  }

  return jackpot_msg;
}

bool check_jackpot_words(const char* tag, const std::array<uint32_t, 16>& got, const std::array<uint32_t, 16>& want)
{
  if (got == want)
  {
    TNN_LOG_INFO("%s vector.jackpot_words OK\n", tag);
    return true;
  }

  TNN_LOG_ERROR("%s vector.jackpot_words mismatch\n", tag);
  for (size_t i = 0; i < got.size(); ++i)
  {
    if (got[i] != want[i])
    {
      TNN_LOG_ERROR("%s   word[%zu] got=%08x want=%08x\n", tag, i, got[i], want[i]);
    }
  }
  return false;
}

#ifdef TNN_HIP
bool pearl_gpu_check(const char* tag, oroError_t err, const char* expr, const char* file, int line)
{
  if (err == oroSuccess)
    return true;

  TNN_LOG_ERROR("%s GPU call failed: %s err=%d (%s) at %s:%d\n",
                tag,
                expr,
                static_cast<int>(err),
                tnn_error_string(err),
                file,
                line);
  return false;
}

#define PEARL_GPU_CHECK(call) pearl_gpu_check(tag, (call), #call, __FILE__, __LINE__)

std::array<uint8_t, 32> pearl_noise_seed_label(const char* label)
{
  std::array<uint8_t, 32> out{};
  std::copy(label, label + std::strlen(label), out.begin());
  return out;
}

int8_t pearl_dense_noise_byte(uint8_t raw)
{
  constexpr int noise_abs_max = 128;
  constexpr int noise_range = 64;
  const int32_t signed_raw = static_cast<int32_t>(static_cast<int8_t>(raw));
  return static_cast<int8_t>(((signed_raw + noise_abs_max) % noise_range) - (noise_range / 2));
}

std::vector<int8_t> pearl_dense_noise_reference(
  size_t num_rows,
  size_t cols,
  const Hash256& key,
  const std::array<uint8_t, 32>& seed)
{
  constexpr size_t digest_bytes = 32;
  std::vector<int8_t> out(num_rows * cols);
  const size_t messages = (out.size() + digest_bytes - 1) / digest_bytes;

  for (size_t message_index = 0; message_index < messages; ++message_index)
  {
    std::array<uint8_t, 64> message{};
    const uint32_t thread_coord = static_cast<uint32_t>(message_index + 1);
    message[0] = static_cast<uint8_t>(thread_coord & 0xff);
    message[1] = static_cast<uint8_t>((thread_coord >> 8) & 0xff);
    message[2] = static_cast<uint8_t>((thread_coord >> 16) & 0xff);
    message[3] = static_cast<uint8_t>((thread_coord >> 24) & 0xff);
    std::copy(seed.begin(), seed.end(), message.begin() + 32);

    const Hash256 digest = blake3_keyed_digest(message.data(), message.size(), key);
    for (size_t j = 0; j < digest_bytes; ++j)
    {
      const size_t out_index = message_index * digest_bytes + j;
      if (out_index < out.size())
        out[out_index] = pearl_dense_noise_byte(digest[j]);
    }
  }

  return out;
}

std::vector<int8_t> pearl_sparse_noise_reference(
  size_t num_rows,
  size_t cols,
  const Hash256& key,
  const std::array<uint8_t, 32>& seed)
{
  constexpr size_t digest_words = 8;
  std::vector<int8_t> out(num_rows * cols, 0);
  const size_t messages = (num_rows + digest_words - 1) / digest_words;

  for (size_t message_index = 0; message_index < messages; ++message_index)
  {
    std::array<uint8_t, 64> message{};
    const uint32_t thread_coord = static_cast<uint32_t>(message_index + 1);
    message[4] = static_cast<uint8_t>(thread_coord & 0xff);
    message[5] = static_cast<uint8_t>((thread_coord >> 8) & 0xff);
    message[6] = static_cast<uint8_t>((thread_coord >> 16) & 0xff);
    message[7] = static_cast<uint8_t>((thread_coord >> 24) & 0xff);
    std::copy(seed.begin(), seed.end(), message.begin() + 32);

    const Hash256 digest = blake3_keyed_digest(message.data(), message.size(), key);
    for (size_t j = 0; j < digest_words; ++j)
    {
      const size_t row = message_index * digest_words + j;
      if (row >= num_rows)
        break;

      const uint32_t u =
        static_cast<uint32_t>(digest[j * 4]) |
        (static_cast<uint32_t>(digest[j * 4 + 1]) << 8) |
        (static_cast<uint32_t>(digest[j * 4 + 2]) << 16) |
        (static_cast<uint32_t>(digest[j * 4 + 3]) << 24);
      const size_t r0 = u & static_cast<uint32_t>(cols - 1);
      const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(cols - 1) * u) >> 32);
      const size_t r1 = r0 ^ static_cast<size_t>(1 + hi);
      out[row * cols + r0] = 1;
      out[row * cols + r1] = -1;
    }
  }

  return out;
}

std::string rtc_include_basename(const std::string& include_name)
{
  const auto pos = include_name.find_last_of("/\\");
  if (pos == std::string::npos)
    return include_name;
  return include_name.substr(pos + 1);
}

std::vector<std::string> pearl_rtc_compile_opts(const oroDeviceProp_t& props, bool is_amd)
{
  std::vector<std::string> opts;
  if (is_amd)
  {
    opts = {"-O3", "-mno-cumode", "-ffast-math", "-D__HIPCC_RTC__", "-D__HIPRTC__"};
    if (props.gcnArchName[0] != '\0')
      opts.push_back(std::string("--gpu-architecture=") + props.gcnArchName);
  }
  else
  {
    opts = {"--dopt=on", "--use_fast_math"};
#ifdef __linux__
    opts.push_back("--device-int128");
#endif
    opts.push_back("-D__CUDACC_RTC__");
    char arch_buf[32];
    std::snprintf(arch_buf, sizeof(arch_buf), "sm_%d%d", props.major, props.minor);
    opts.push_back(std::string("--gpu-architecture=") + arch_buf);
  }
  return opts;
}

bool run_dense_noise_gpu_case(
  const char* tag,
  const char* name,
  oroFunction_t kernel,
  int num_rows,
  int cols,
  const Hash256& key,
  const std::array<uint8_t, 32>& seed)
{
  constexpr int threads = 128;
  constexpr int rows_per_block = 128;

  const int blocks = (num_rows + rows_per_block - 1) / rows_per_block;
  const size_t out_bytes = static_cast<size_t>(num_rows) * cols;

  int8_t* d_out = nullptr;
  uint8_t* d_key = nullptr;
  uint8_t* d_seed = nullptr;

  if (!PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_out), out_bytes)) ||
      !PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_key), key.size())) ||
      !PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_seed), seed.size())))
  {
    return false;
  }

  bool ok = true;
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_key, key.data(), key.size(), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_seed, seed.data(), seed.size(), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemset(d_out, 0, out_bytes));

  uint32_t thread_coord_base = 0;
  void* args[] = {&d_out, &num_rows, &d_key, &d_seed, &thread_coord_base};
  ok &= PEARL_GPU_CHECK(oroModuleLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
  ok &= PEARL_GPU_CHECK(oroDeviceSynchronize());

  std::vector<int8_t> got(out_bytes);
  ok &= PEARL_GPU_CHECK(oroMemcpy(got.data(), d_out, out_bytes, oroMemcpyDeviceToHost));

  (void)oroFree(reinterpret_cast<oroDeviceptr>(d_out));
  (void)oroFree(reinterpret_cast<oroDeviceptr>(d_key));
  (void)oroFree(reinterpret_cast<oroDeviceptr>(d_seed));

  if (!ok)
    return false;

  const std::vector<int8_t> want = pearl_dense_noise_reference(num_rows, cols, key, seed);
  if (got == want)
  {
    TNN_LOG_INFO("%s gpu.%s_dense OK\n", tag, name);
    return true;
  }

  TNN_LOG_ERROR("%s gpu.%s_dense mismatch\n", tag, name);
  for (size_t i = 0; i < got.size(); ++i)
  {
    if (got[i] != want[i])
    {
      TNN_LOG_ERROR("%s   first mismatch row=%zu col=%zu got=%d want=%d\n",
                    tag,
                    i / static_cast<size_t>(cols),
                    i % static_cast<size_t>(cols),
                    static_cast<int>(got[i]),
                    static_cast<int>(want[i]));
      break;
    }
  }
  return false;
}

bool run_sparse_noise_gpu_case(
  const char* tag,
  const char* name,
  oroFunction_t kernel,
  int num_rows,
  int cols,
  const Hash256& key,
  const std::array<uint8_t, 32>& seed)
{
  constexpr int threads = 128;
  constexpr int rows_per_thread = 8;

  const int blocks = (num_rows + threads * rows_per_thread - 1) / (threads * rows_per_thread);
  const size_t out_bytes = static_cast<size_t>(num_rows) * cols;

  int8_t* d_out = nullptr;
  uint8_t* d_key = nullptr;
  uint8_t* d_seed = nullptr;

  if (!PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_out), out_bytes)) ||
      !PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_key), key.size())) ||
      !PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_seed), seed.size())))
  {
    return false;
  }

  bool ok = true;
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_key, key.data(), key.size(), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_seed, seed.data(), seed.size(), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemset(d_out, 0, out_bytes));

  uint32_t thread_coord_base = 0;
  void* args[] = {&d_out, &num_rows, &d_key, &d_seed, &thread_coord_base};
  ok &= PEARL_GPU_CHECK(oroModuleLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
  ok &= PEARL_GPU_CHECK(oroDeviceSynchronize());

  std::vector<int8_t> got(out_bytes);
  ok &= PEARL_GPU_CHECK(oroMemcpy(got.data(), d_out, out_bytes, oroMemcpyDeviceToHost));

  (void)oroFree(reinterpret_cast<oroDeviceptr>(d_out));
  (void)oroFree(reinterpret_cast<oroDeviceptr>(d_key));
  (void)oroFree(reinterpret_cast<oroDeviceptr>(d_seed));

  if (!ok)
    return false;

  const std::vector<int8_t> want = pearl_sparse_noise_reference(num_rows, cols, key, seed);
  if (got == want)
  {
    TNN_LOG_INFO("%s gpu.%s_sparse OK\n", tag, name);
    return true;
  }

  TNN_LOG_ERROR("%s gpu.%s_sparse mismatch\n", tag, name);
  for (size_t i = 0; i < got.size(); ++i)
  {
    if (got[i] != want[i])
    {
      TNN_LOG_ERROR("%s   first mismatch row=%zu col=%zu got=%d want=%d\n",
                    tag,
                    i / static_cast<size_t>(cols),
                    i % static_cast<size_t>(cols),
                    static_cast<int>(got[i]),
                    static_cast<int>(want[i]));
      break;
    }
  }
  return false;
}

bool run_noise_gpu_checks(
  const char* tag,
  const HarnessFixture& fixture,
  const Hash256& a_noise_seed,
  const Hash256& b_noise_seed)
{
  int device_count = 0;
  if (!PEARL_GPU_CHECK(oroGetDeviceCount(&device_count)))
    return false;
  if (device_count == 0)
  {
    TNN_LOG_ERROR("%s no GPU devices found for noise checks\n", tag);
    return false;
  }

  oroDeviceProp_t props{};
  if (!PEARL_GPU_CHECK(oroGetDeviceProperties(&props, tnn_get_device(0))))
    return false;

  oroCtx ctx{};
  if (!PEARL_GPU_CHECK(oroCtxCreate(&ctx, 0, tnn_get_device(0))))
    return false;

  auto& compiler = RTCCompiler::instance();
  auto rtc_headers = build_rtc_headers(
    hip_embedded::PEARL_HEADERS,
    hip_embedded::IRIS_HEADERS,
    hip_embedded::COMMON_HEADERS);
  for (const auto& h : rtc_headers)
  {
    const std::string include_name(h.name);
    const std::string source(h.source);
    compiler.add_header_source(include_name, source);

    const std::string basename = rtc_include_basename(include_name);
    if (basename != include_name)
      compiler.add_header_source(basename, source);
  }

  const bool is_amd = tnn_is_amd_device(0);
  const auto compile_opts = pearl_rtc_compile_opts(props, is_amd);
  const std::string source(
    hip_pearl_noise_dense_source::SRC_TNN_HIP_CRYPTO_PEARL_NOISE_GENERATION_DENSE_TEST_HIP_SOURCE);

  TNN_LOG_INFO("%s compiling Pearl noise HIPRTC kernels\n", tag);
  RTCCompiler::CompiledKernel dense_compiled{};
  RTCCompiler::CompiledKernel sparse_compiled{};
  try
  {
    dense_compiled = compiler.compile_from_source(
      source,
      "pearl-noise-dense-test.hip",
      "pearl_noise_dense_32_128_kernel",
      compile_opts);
    sparse_compiled = compiler.compile_from_source(
      source,
      "pearl-noise-dense-test.hip",
      "pearl_noise_sparse_32_128_kernel",
      compile_opts);
  }
  catch (const std::exception& e)
  {
    TNN_LOG_ERROR("%s Pearl noise HIPRTC compile failed: %s\n", tag, e.what());
    (void)oroCtxDestroy(ctx);
    return false;
  }

  const auto seed_a = pearl_noise_seed_label("A_tensor");
  const auto seed_b = pearl_noise_seed_label("B_tensor");
  const int dense_cols = static_cast<int>(fixture.config.rank);
  if (dense_cols != 32)
  {
    TNN_LOG_ERROR("%s dense noise HIP test currently expects rank=32, got %d\n", tag, dense_cols);
    (void)oroCtxDestroy(ctx);
    return false;
  }

  bool ok = true;
  ok &= run_dense_noise_gpu_case(
    tag, "noise_a_factor", dense_compiled.function, static_cast<int>(fixture.m), dense_cols, a_noise_seed, seed_a);
  ok &= run_dense_noise_gpu_case(
    tag, "noise_b_factor", dense_compiled.function, static_cast<int>(fixture.n), dense_cols, b_noise_seed, seed_b);
  ok &= run_sparse_noise_gpu_case(
    tag, "sparse_a_factor", sparse_compiled.function, static_cast<int>(fixture.config.common_dim), dense_cols, a_noise_seed, seed_a);
  ok &= run_sparse_noise_gpu_case(
    tag, "sparse_b_factor", sparse_compiled.function, static_cast<int>(fixture.config.common_dim), dense_cols, b_noise_seed, seed_b);

  (void)oroCtxDestroy(ctx);
  return ok;
}

// Helper constants for jackpot GPU kernel
static constexpr int kNumThreads = 256;
static constexpr int kJackpotSize = 16;

bool run_jackpot_gpu_check(
  const char* tag,
  const I32Matrix& s_a,
  const I32Matrix& s_b,
  const I32Matrix& noise_a,
  const I32Matrix& noise_b,
  int k_in,
  int rank,
  const std::array<uint32_t, 16>& expected_jackpot)
{
  int h = static_cast<int>(s_a.size());
  int w = static_cast<int>(s_b.size());
  int k = k_in;
  const int tiles = k / rank;
  const int jackpot_tile_rows_a = 16;
  const int jackpot_tile_rows_b = 32;
  const unsigned int partial_grid_x = static_cast<unsigned int>((w + jackpot_tile_rows_b - 1) / jackpot_tile_rows_b);
  const unsigned int partial_grid_y = static_cast<unsigned int>((h + jackpot_tile_rows_a - 1) / jackpot_tile_rows_a);
  int partial_blocks = static_cast<int>(partial_grid_x * partial_grid_y);
  const size_t partial_bytes = static_cast<size_t>(partial_blocks) * tiles * sizeof(uint32_t);

  // Convert matrices from I32Matrix (int32) to flat int8 arrays
  const size_t a_bytes = static_cast<size_t>(h) * k;
  const size_t b_bytes = static_cast<size_t>(w) * k;
  std::vector<int8_t> s_a_flat(a_bytes);
  std::vector<int8_t> noise_a_flat(a_bytes);
  std::vector<int8_t> s_b_flat(b_bytes);
  std::vector<int8_t> noise_b_flat(b_bytes);

  for (int i = 0; i < h; ++i)
    for (int j = 0; j < k; ++j)
      s_a_flat[i * k + j] = static_cast<int8_t>(s_a[i][j]);
  for (int i = 0; i < h; ++i)
    for (int j = 0; j < k; ++j)
      noise_a_flat[i * k + j] = static_cast<int8_t>(noise_a[i][j]);
  for (int i = 0; i < w; ++i)
    for (int j = 0; j < k; ++j)
      s_b_flat[i * k + j] = static_cast<int8_t>(s_b[i][j]);
  for (int i = 0; i < w; ++i)
    for (int j = 0; j < k; ++j)
      noise_b_flat[i * k + j] = static_cast<int8_t>(noise_b[i][j]);

  int device_count = 0;
  if (!PEARL_GPU_CHECK(oroGetDeviceCount(&device_count)))
    return false;
  if (device_count == 0)
  {
    TNN_LOG_ERROR("%s no GPU devices found for jackpot check\n", tag);
    return false;
  }

  oroDeviceProp_t props{};
  if (!PEARL_GPU_CHECK(oroGetDeviceProperties(&props, tnn_get_device(0))))
    return false;

  oroCtx ctx{};
  if (!PEARL_GPU_CHECK(oroCtxCreate(&ctx, 0, tnn_get_device(0))))
    return false;

  auto& compiler = RTCCompiler::instance();
  auto rtc_headers = build_rtc_headers(
    hip_embedded::PEARL_HEADERS,
    hip_embedded::IRIS_HEADERS,
    hip_embedded::COMMON_HEADERS);
  for (const auto& hdr : rtc_headers)
  {
    const std::string include_name(hdr.name);
    const std::string source(hdr.source);
    compiler.add_header_source(include_name, source);

    const std::string basename = rtc_include_basename(include_name);
    if (basename != include_name)
      compiler.add_header_source(basename, source);
  }

  const bool is_amd = tnn_is_amd_device(0);
  const auto compile_opts = pearl_rtc_compile_opts(props, is_amd);
  const std::string source(
    hip_pearl_expand_jackpot_source::SRC_TNN_HIP_CRYPTO_PEARL_PEARL_EXPAND_JACKPOT_HIP_SOURCE);

  TNN_LOG_INFO("%s compiling Pearl jackpot HIPRTC kernels\n", tag);
  RTCCompiler::CompiledKernel partial_compiled{};
  RTCCompiler::CompiledKernel reduce_compiled{};
  try
  {
    partial_compiled = compiler.compile_from_source(
      source,
      "pearl-expand-jackpot.hip",
      "pearl_expand_jackpot_tiled_partial",
      compile_opts);
    reduce_compiled = compiler.compile_from_source(
      source,
      "pearl-expand-jackpot.hip",
      "pearl_jackpot_tiled_reduce",
      compile_opts);
  }
  catch (const std::exception& e)
  {
    TNN_LOG_ERROR("%s Pearl jackpot HIPRTC compile failed: %s\n", tag, e.what());
    (void)oroCtxDestroy(ctx);
    return false;
  }

  // Allocate GPU memory
  signed char* d_s_a = nullptr;
  signed char* d_noise_a = nullptr;
  signed char* d_s_b = nullptr;
  signed char* d_noise_b = nullptr;
  uint32_t* d_partial = nullptr;
  uint32_t* d_jackpot = nullptr;

  bool alloc_ok = true;
  alloc_ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_s_a), a_bytes));
  alloc_ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_noise_a), a_bytes));
  alloc_ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_s_b), b_bytes));
  alloc_ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_noise_b), b_bytes));
  alloc_ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_partial), partial_bytes));
  alloc_ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_jackpot), kJackpotSize * sizeof(uint32_t)));

  if (!alloc_ok)
  {
    auto free_all = [&]() {
      if (d_s_a)     (void)oroFree(reinterpret_cast<oroDeviceptr>(d_s_a));
      if (d_noise_a) (void)oroFree(reinterpret_cast<oroDeviceptr>(d_noise_a));
      if (d_s_b)     (void)oroFree(reinterpret_cast<oroDeviceptr>(d_s_b));
      if (d_noise_b) (void)oroFree(reinterpret_cast<oroDeviceptr>(d_noise_b));
      if (d_partial)  (void)oroFree(reinterpret_cast<oroDeviceptr>(d_partial));
      if (d_jackpot)  (void)oroFree(reinterpret_cast<oroDeviceptr>(d_jackpot));
    };
    free_all();
    (void)oroCtxDestroy(ctx);
    return false;
  }

  bool ok = true;
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_s_a, s_a_flat.data(), a_bytes, oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_noise_a, noise_a_flat.data(), a_bytes, oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_s_b, s_b_flat.data(), b_bytes, oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_noise_b, noise_b_flat.data(), b_bytes, oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemset(d_partial, 0, partial_bytes));
  ok &= PEARL_GPU_CHECK(oroMemset(d_jackpot, 0, kJackpotSize * sizeof(uint32_t)));

  if (!ok)
  {
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_s_a));
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_noise_a));
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_s_b));
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_noise_b));
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_partial));
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_jackpot));
    (void)oroCtxDestroy(ctx);
    return false;
  }

  // Launch Phase 1: partial jackpot computation
  void* partial_args[] = {&d_s_a, &d_noise_a, &d_s_b, &d_noise_b, &d_partial, &h, &w, &k};
  ok &= PEARL_GPU_CHECK(oroModuleLaunchKernel(
    partial_compiled.function, partial_grid_x, partial_grid_y, 1, kNumThreads, 1, 1, 0, nullptr, partial_args, nullptr));

  if (!ok)
  {
    TNN_LOG_ERROR("%s pearl_expand_jackpot_tiled_partial launch failed\n", tag);
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_s_a));
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_noise_a));
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_s_b));
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_noise_b));
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_partial));
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_jackpot));
    (void)oroCtxDestroy(ctx);
    return false;
  }

  // Launch Phase 2: reduction
  void* reduce_args[] = {&d_partial, &d_jackpot, &partial_blocks, &k};
  ok &= PEARL_GPU_CHECK(oroModuleLaunchKernel(
    reduce_compiled.function, kJackpotSize, 1, 1, kNumThreads, 1, 1, 0, nullptr, reduce_args, nullptr));

  ok &= PEARL_GPU_CHECK(oroDeviceSynchronize());

  if (!ok)
  {
    TNN_LOG_ERROR("%s pearl_jackpot_tiled_reduce launch or sync failed\n", tag);
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_s_a));
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_noise_a));
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_s_b));
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_noise_b));
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_partial));
    (void)oroFree(reinterpret_cast<oroDeviceptr>(d_jackpot));
    (void)oroCtxDestroy(ctx);
    return false;
  }

  // Download result
  std::array<uint32_t, kJackpotSize> got_jackpot{};
  ok &= PEARL_GPU_CHECK(oroMemcpy(
    got_jackpot.data(), d_jackpot, kJackpotSize * sizeof(uint32_t), oroMemcpyDeviceToHost));

  // Cleanup
  (void)oroFree(reinterpret_cast<oroDeviceptr>(d_s_a));
  (void)oroFree(reinterpret_cast<oroDeviceptr>(d_noise_a));
  (void)oroFree(reinterpret_cast<oroDeviceptr>(d_s_b));
  (void)oroFree(reinterpret_cast<oroDeviceptr>(d_noise_b));
  (void)oroFree(reinterpret_cast<oroDeviceptr>(d_partial));
  (void)oroFree(reinterpret_cast<oroDeviceptr>(d_jackpot));
  (void)oroCtxDestroy(ctx);

  if (!ok)
    return false;

  // Compare
  if (got_jackpot == expected_jackpot)
  {
    TNN_LOG_INFO("%s gpu.jackpot_words OK\n", tag);
    return true;
  }

  TNN_LOG_ERROR("%s gpu.jackpot_words mismatch\n", tag);
  for (size_t i = 0; i < got_jackpot.size(); ++i)
  {
    if (got_jackpot[i] != expected_jackpot[i])
    {
      TNN_LOG_ERROR("%s   word[%zu] got=%08x want=%08x\n", tag, i, got_jackpot[i], expected_jackpot[i]);
    }
  }
  return false;
}

float pearl_time_kernel_ms(
  const char* tag,
  const char* name,
  oroFunction_t kernel,
  unsigned int grid_x,
  unsigned int grid_y,
  unsigned int grid_z,
  unsigned int block_x,
  void** args,
  int warmup,
  int iterations)
{
  for (int i = 0; i < warmup; ++i)
  {
    if (!PEARL_GPU_CHECK(oroModuleLaunchKernel(kernel, grid_x, grid_y, grid_z, block_x, 1, 1, 0, nullptr, args, nullptr)))
      return -1.0f;
  }
  if (!PEARL_GPU_CHECK(oroDeviceSynchronize()))
    return -1.0f;

  oroEvent_t start = nullptr;
  oroEvent_t stop = nullptr;
  if (!PEARL_GPU_CHECK(oroEventCreate(&start)) ||
      !PEARL_GPU_CHECK(oroEventCreate(&stop)))
  {
    if (start) (void)oroEventDestroy(start);
    if (stop) (void)oroEventDestroy(stop);
    return -1.0f;
  }

  bool ok = true;
  ok &= PEARL_GPU_CHECK(oroEventRecord(start, nullptr));
  for (int i = 0; i < iterations; ++i)
  {
    ok &= PEARL_GPU_CHECK(oroModuleLaunchKernel(kernel, grid_x, grid_y, grid_z, block_x, 1, 1, 0, nullptr, args, nullptr));
  }
  ok &= PEARL_GPU_CHECK(oroEventRecord(stop, nullptr));
  ok &= PEARL_GPU_CHECK(oroEventSynchronize(stop));

  float total_ms = 0.0f;
  ok &= PEARL_GPU_CHECK(oroEventElapsedTime(&total_ms, start, stop));
  (void)oroEventDestroy(start);
  (void)oroEventDestroy(stop);

  if (!ok)
    return -1.0f;

  const float avg_ms = total_ms / static_cast<float>(iterations);
  TNN_LOG_INFO("%s %-22s avg=%.4f ms iterations=%d\n", tag, name, avg_ms, iterations);
  return avg_ms;
}

std::vector<int8_t> pearl_bench_matrix(size_t size, uint32_t seed)
{
  std::vector<int8_t> out(size);
  uint32_t x = seed;
  for (size_t i = 0; i < size; ++i)
  {
    x = x * 1664525u + 1013904223u;
    out[i] = static_cast<int8_t>(static_cast<int>((x >> 24) & 63u) - 32);
  }
  return out;
}

bool pearl_register_rtc_headers()
{
  auto& compiler = RTCCompiler::instance();
  auto rtc_headers = build_rtc_headers(
    hip_embedded::PEARL_HEADERS,
    hip_embedded::IRIS_HEADERS,
    hip_embedded::COMMON_HEADERS);
  for (const auto& hdr : rtc_headers)
  {
    const std::string include_name(hdr.name);
    const std::string source(hdr.source);
    compiler.add_header_source(include_name, source);

    const std::string basename = rtc_include_basename(include_name);
    if (basename != include_name)
      compiler.add_header_source(basename, source);
  }
  return true;
}

int run_pearl_benchmark(const char* tag)
{
  constexpr int m = 1024;
  constexpr int n = 1024;
  constexpr int k = 4096;
  constexpr int rank = 32;
  constexpr int noise_threads = 128;
  constexpr int jackpot_threads = 256;
  constexpr int warmup = 3;
  constexpr int noise_iterations = 200;
  constexpr int jackpot_iterations = 10;

  int device_count = 0;
  if (!PEARL_GPU_CHECK(oroGetDeviceCount(&device_count)))
    return 1;
  if (device_count == 0)
  {
    TNN_LOG_ERROR("%s no GPU devices found for benchmark\n", tag);
    return 1;
  }

  oroDeviceProp_t props{};
  if (!PEARL_GPU_CHECK(oroGetDeviceProperties(&props, tnn_get_device(0))))
    return 1;

  oroCtx ctx{};
  if (!PEARL_GPU_CHECK(oroCtxCreate(&ctx, 0, tnn_get_device(0))))
    return 1;

  pearl_register_rtc_headers();
  auto& compiler = RTCCompiler::instance();
  const bool is_amd = tnn_is_amd_device(0);
  const auto compile_opts = pearl_rtc_compile_opts(props, is_amd);
  const std::string noise_source(
    hip_pearl_noise_dense_source::SRC_TNN_HIP_CRYPTO_PEARL_NOISE_GENERATION_DENSE_TEST_HIP_SOURCE);
  const std::string jackpot_source(
    hip_pearl_expand_jackpot_source::SRC_TNN_HIP_CRYPTO_PEARL_PEARL_EXPAND_JACKPOT_HIP_SOURCE);

  RTCCompiler::CompiledKernel dense_compiled{};
  RTCCompiler::CompiledKernel sparse_compiled{};
  RTCCompiler::CompiledKernel partial_compiled{};
  RTCCompiler::CompiledKernel wmma_partial_compiled{};
  RTCCompiler::CompiledKernel reduce_compiled{};
  try
  {
    dense_compiled = compiler.compile_from_source(
      noise_source, "pearl-noise-dense-test.hip", "pearl_noise_dense_32_128_kernel", compile_opts);
    sparse_compiled = compiler.compile_from_source(
      noise_source, "pearl-noise-dense-test.hip", "pearl_noise_sparse_32_128_kernel", compile_opts);
    partial_compiled = compiler.compile_from_source(
      jackpot_source, "pearl-expand-jackpot.hip", "pearl_expand_jackpot_tiled_partial", compile_opts);
    wmma_partial_compiled = compiler.compile_from_source(
      jackpot_source, "pearl-expand-jackpot.hip", "pearl_expand_jackpot_wmma_partial", compile_opts);
    reduce_compiled = compiler.compile_from_source(
      jackpot_source, "pearl-expand-jackpot.hip", "pearl_jackpot_tiled_reduce", compile_opts);
  }
  catch (const std::exception& e)
  {
    TNN_LOG_ERROR("%s Pearl benchmark HIPRTC compile failed: %s\n", tag, e.what());
    (void)oroCtxDestroy(ctx);
    return 1;
  }

  const size_t dense_a_bytes = static_cast<size_t>(m) * rank;
  const size_t dense_b_bytes = static_cast<size_t>(n) * rank;
  const size_t sparse_bytes = static_cast<size_t>(k) * rank;
  const size_t matrix_a_bytes = static_cast<size_t>(m) * k;
  const size_t matrix_b_bytes = static_cast<size_t>(n) * k;
  const int tiles = k / rank;
  const int jackpot_tile_rows_a = 16;
  const int jackpot_tile_rows_b = 32;
  const int jackpot_wmma_tile_rows_a = 64;
  const int jackpot_wmma_tile_rows_b = 32;
  const unsigned int partial_grid_x = static_cast<unsigned int>((n + jackpot_tile_rows_b - 1) / jackpot_tile_rows_b);
  const unsigned int partial_grid_y = static_cast<unsigned int>((m + jackpot_tile_rows_a - 1) / jackpot_tile_rows_a);
  int partial_blocks = static_cast<int>(partial_grid_x * partial_grid_y);
  const unsigned int wmma_partial_grid_x = static_cast<unsigned int>((n + jackpot_wmma_tile_rows_b - 1) / jackpot_wmma_tile_rows_b);
  const unsigned int wmma_partial_grid_y = static_cast<unsigned int>((m + jackpot_wmma_tile_rows_a - 1) / jackpot_wmma_tile_rows_a);
  int wmma_partial_blocks = static_cast<int>(wmma_partial_grid_x * wmma_partial_grid_y);
  constexpr int jackpot_batch_probe = 8;
  const size_t partial_bytes = static_cast<size_t>(partial_blocks) * tiles * sizeof(uint32_t);
  const size_t partial_alloc_bytes = partial_bytes * jackpot_batch_probe;

  const auto seed_a = pearl_noise_seed_label("A_tensor");
  const auto seed_b = pearl_noise_seed_label("B_tensor");
  Hash256 key_a{};
  Hash256 key_b{};
  for (size_t i = 0; i < key_a.size(); ++i)
  {
    key_a[i] = static_cast<uint8_t>(0x11u + i * 7u);
    key_b[i] = static_cast<uint8_t>(0xc2u + i * 5u);
  }

  std::vector<int8_t> s_a = pearl_bench_matrix(matrix_a_bytes, 0x12345678u);
  std::vector<int8_t> s_b = pearl_bench_matrix(matrix_b_bytes, 0x9abcdef0u);
  std::vector<int8_t> noise_a = pearl_bench_matrix(matrix_a_bytes, 0x0badc0deu);
  std::vector<int8_t> noise_b = pearl_bench_matrix(matrix_b_bytes, 0xfeedfaceu);

  int8_t* d_dense_a = nullptr;
  int8_t* d_dense_b = nullptr;
  int8_t* d_sparse = nullptr;
  uint8_t* d_key_a = nullptr;
  uint8_t* d_key_b = nullptr;
  uint8_t* d_seed_a = nullptr;
  uint8_t* d_seed_b = nullptr;
  int8_t* d_s_a = nullptr;
  int8_t* d_s_b = nullptr;
  int8_t* d_noise_a = nullptr;
  int8_t* d_noise_b = nullptr;
  uint32_t* d_partial = nullptr;
  uint32_t* d_jackpot = nullptr;

  bool ok = true;
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_dense_a), dense_a_bytes));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_dense_b), dense_b_bytes));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_sparse), sparse_bytes));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_key_a), key_a.size()));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_key_b), key_b.size()));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_seed_a), seed_a.size()));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_seed_b), seed_b.size()));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_s_a), matrix_a_bytes));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_s_b), matrix_b_bytes));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_noise_a), matrix_a_bytes));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_noise_b), matrix_b_bytes));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_partial), partial_alloc_bytes));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_jackpot), kJackpotSize * sizeof(uint32_t)));

  if (ok)
  {
    ok &= PEARL_GPU_CHECK(oroMemcpy(d_key_a, key_a.data(), key_a.size(), oroMemcpyHostToDevice));
    ok &= PEARL_GPU_CHECK(oroMemcpy(d_key_b, key_b.data(), key_b.size(), oroMemcpyHostToDevice));
    ok &= PEARL_GPU_CHECK(oroMemcpy(d_seed_a, seed_a.data(), seed_a.size(), oroMemcpyHostToDevice));
    ok &= PEARL_GPU_CHECK(oroMemcpy(d_seed_b, seed_b.data(), seed_b.size(), oroMemcpyHostToDevice));
    ok &= PEARL_GPU_CHECK(oroMemcpy(d_s_a, s_a.data(), matrix_a_bytes, oroMemcpyHostToDevice));
    ok &= PEARL_GPU_CHECK(oroMemcpy(d_s_b, s_b.data(), matrix_b_bytes, oroMemcpyHostToDevice));
    ok &= PEARL_GPU_CHECK(oroMemcpy(d_noise_a, noise_a.data(), matrix_a_bytes, oroMemcpyHostToDevice));
    ok &= PEARL_GPU_CHECK(oroMemcpy(d_noise_b, noise_b.data(), matrix_b_bytes, oroMemcpyHostToDevice));
    ok &= PEARL_GPU_CHECK(oroMemset(d_sparse, 0, sparse_bytes));
    ok &= PEARL_GPU_CHECK(oroMemset(d_partial, 0, partial_alloc_bytes));
    ok &= PEARL_GPU_CHECK(oroMemset(d_jackpot, 0, kJackpotSize * sizeof(uint32_t)));
  }

  if (!ok)
  {
    TNN_LOG_ERROR("%s setup failed\n", tag);
  }
  else
  {
    TNN_LOG_INFO_COLOR(BRIGHT_CYAN, "%s Pearl PoUW realistic GPU benchmark\n", tag);
    TNN_LOG_INFO("%s shape m=%d n=%d k=%d rank=%d tiles=%d\n", tag, m, n, k, rank, tiles);
    TNN_LOG_INFO("%s warmup=%d noise_iters=%d jackpot_iters=%d\n", tag, warmup, noise_iterations, jackpot_iterations);

    int m_arg = m;
    int n_arg = n;
    int k_arg = k;
    uint32_t coord_base = 0;
    void* dense_a_args[] = {&d_dense_a, &m_arg, &d_key_a, &d_seed_a, &coord_base};
    void* dense_b_args[] = {&d_dense_b, &n_arg, &d_key_b, &d_seed_b, &coord_base};
    void* sparse_a_args[] = {&d_sparse, &k_arg, &d_key_a, &d_seed_a, &coord_base};
    void* sparse_b_args[] = {&d_sparse, &k_arg, &d_key_b, &d_seed_b, &coord_base};
    void* partial_args[] = {&d_s_a, &d_noise_a, &d_s_b, &d_noise_b, &d_partial, &m_arg, &n_arg, &k_arg};
    void* reduce_args[] = {&d_partial, &d_jackpot, &partial_blocks, &k_arg};
    void* wmma_reduce_args[] = {&d_partial, &d_jackpot, &wmma_partial_blocks, &k_arg};

    const unsigned dense_a_grid = static_cast<unsigned>((m + 127) / 128);
    const unsigned dense_b_grid = static_cast<unsigned>((n + 127) / 128);
    const unsigned sparse_grid = static_cast<unsigned>((k + noise_threads * 8 - 1) / (noise_threads * 8));

    const float dense_a_ms = pearl_time_kernel_ms(tag, "dense A factor", dense_compiled.function, dense_a_grid, 1, 1, noise_threads, dense_a_args, warmup, noise_iterations);
    const float dense_b_ms = pearl_time_kernel_ms(tag, "dense B factor", dense_compiled.function, dense_b_grid, 1, 1, noise_threads, dense_b_args, warmup, noise_iterations);
    const float sparse_a_ms = pearl_time_kernel_ms(tag, "sparse A factor", sparse_compiled.function, sparse_grid, 1, 1, noise_threads, sparse_a_args, warmup, noise_iterations);
    const float sparse_b_ms = pearl_time_kernel_ms(tag, "sparse B factor", sparse_compiled.function, sparse_grid, 1, 1, noise_threads, sparse_b_args, warmup, noise_iterations);

    (void)oroMemset(d_partial, 0, partial_alloc_bytes);
    (void)oroMemset(d_jackpot, 0, kJackpotSize * sizeof(uint32_t));
    const float partial_ms = pearl_time_kernel_ms(tag, "jackpot partial", partial_compiled.function, partial_grid_x, partial_grid_y, 1, jackpot_threads, partial_args, warmup, jackpot_iterations);
    const float reduce_ms = pearl_time_kernel_ms(tag, "jackpot reduce", reduce_compiled.function, kJackpotSize, 1, 1, jackpot_threads, reduce_args, warmup, jackpot_iterations);
    const float partial_batch2_ms = pearl_time_kernel_ms(tag, "jackpot partial x2", partial_compiled.function, partial_grid_x, partial_grid_y, 2, jackpot_threads, partial_args, warmup, jackpot_iterations);
    const float partial_batch4_ms = pearl_time_kernel_ms(tag, "jackpot partial x4", partial_compiled.function, partial_grid_x, partial_grid_y, 4, jackpot_threads, partial_args, warmup, jackpot_iterations);
    (void)oroMemset(d_partial, 0, partial_alloc_bytes);
    (void)oroMemset(d_jackpot, 0, kJackpotSize * sizeof(uint32_t));
    const float wmma_partial_ms = pearl_time_kernel_ms(tag, "jackpot wmma partial", wmma_partial_compiled.function, wmma_partial_grid_x, wmma_partial_grid_y, 1, jackpot_threads, partial_args, warmup, jackpot_iterations);
    const float wmma_reduce_ms = pearl_time_kernel_ms(tag, "jackpot wmma reduce", reduce_compiled.function, kJackpotSize, 1, 1, jackpot_threads, wmma_reduce_args, warmup, jackpot_iterations);
    const float wmma_partial_batch2_ms = pearl_time_kernel_ms(tag, "jackpot wmma partial x2", wmma_partial_compiled.function, wmma_partial_grid_x, wmma_partial_grid_y, 2, jackpot_threads, partial_args, warmup, jackpot_iterations);
    const float wmma_partial_batch4_ms = pearl_time_kernel_ms(tag, "jackpot wmma partial x4", wmma_partial_compiled.function, wmma_partial_grid_x, wmma_partial_grid_y, 4, jackpot_threads, partial_args, warmup, jackpot_iterations);
    const float wmma_partial_batch8_ms = pearl_time_kernel_ms(tag, "jackpot wmma partial x8", wmma_partial_compiled.function, wmma_partial_grid_x, wmma_partial_grid_y, 8, jackpot_threads, partial_args, warmup, jackpot_iterations);

    const double dense_rows = static_cast<double>(m + n);
    const double sparse_rows = static_cast<double>(k * 2);
    const double dot_ops = static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) * 2.0;
    if (dense_a_ms > 0.0f && dense_b_ms > 0.0f)
    {
      const double dense_ms = static_cast<double>(dense_a_ms + dense_b_ms);
      TNN_LOG_INFO("%s dense factors throughput=%.2f Mrows/s\n", tag, dense_rows / dense_ms / 1000.0);
    }
    if (sparse_a_ms > 0.0f && sparse_b_ms > 0.0f)
    {
      const double sparse_ms = static_cast<double>(sparse_a_ms + sparse_b_ms);
      TNN_LOG_INFO("%s sparse factors throughput=%.2f Mrows/s\n", tag, sparse_rows / sparse_ms / 1000.0);
    }
    if (partial_ms > 0.0f)
    {
      TNN_LOG_INFO("%s jackpot partial effective=%.2f TOPS/s\n", tag, dot_ops / (static_cast<double>(partial_ms) * 1.0e9));
    }
    if (partial_batch2_ms > 0.0f)
    {
      TNN_LOG_INFO("%s jackpot partial x2 effective=%.2f TOPS/s\n", tag, (dot_ops * 2.0) / (static_cast<double>(partial_batch2_ms) * 1.0e9));
    }
    if (partial_batch4_ms > 0.0f)
    {
      TNN_LOG_INFO("%s jackpot partial x4 effective=%.2f TOPS/s\n", tag, (dot_ops * 4.0) / (static_cast<double>(partial_batch4_ms) * 1.0e9));
    }
    if (wmma_partial_ms > 0.0f)
    {
      TNN_LOG_INFO("%s jackpot wmma partial effective=%.2f TOPS/s\n", tag, dot_ops / (static_cast<double>(wmma_partial_ms) * 1.0e9));
    }
    if (wmma_partial_batch2_ms > 0.0f)
    {
      TNN_LOG_INFO("%s jackpot wmma partial x2 effective=%.2f TOPS/s\n", tag, (dot_ops * 2.0) / (static_cast<double>(wmma_partial_batch2_ms) * 1.0e9));
    }
    if (wmma_partial_batch4_ms > 0.0f)
    {
      TNN_LOG_INFO("%s jackpot wmma partial x4 effective=%.2f TOPS/s\n", tag, (dot_ops * 4.0) / (static_cast<double>(wmma_partial_batch4_ms) * 1.0e9));
    }
    if (wmma_partial_batch8_ms > 0.0f)
    {
      TNN_LOG_INFO("%s jackpot wmma partial x8 effective=%.2f TOPS/s\n", tag, (dot_ops * 8.0) / (static_cast<double>(wmma_partial_batch8_ms) * 1.0e9));
    }
    if (partial_ms > 0.0f && reduce_ms > 0.0f)
    {
      TNN_LOG_INFO("%s jackpot total avg=%.4f ms\n", tag, partial_ms + reduce_ms);
    }
    if (wmma_partial_ms > 0.0f && wmma_reduce_ms > 0.0f)
    {
      TNN_LOG_INFO("%s jackpot wmma total avg=%.4f ms\n", tag, wmma_partial_ms + wmma_reduce_ms);
    }
  }

  if (d_dense_a) (void)oroFree(reinterpret_cast<oroDeviceptr>(d_dense_a));
  if (d_dense_b) (void)oroFree(reinterpret_cast<oroDeviceptr>(d_dense_b));
  if (d_sparse) (void)oroFree(reinterpret_cast<oroDeviceptr>(d_sparse));
  if (d_key_a) (void)oroFree(reinterpret_cast<oroDeviceptr>(d_key_a));
  if (d_key_b) (void)oroFree(reinterpret_cast<oroDeviceptr>(d_key_b));
  if (d_seed_a) (void)oroFree(reinterpret_cast<oroDeviceptr>(d_seed_a));
  if (d_seed_b) (void)oroFree(reinterpret_cast<oroDeviceptr>(d_seed_b));
  if (d_s_a) (void)oroFree(reinterpret_cast<oroDeviceptr>(d_s_a));
  if (d_s_b) (void)oroFree(reinterpret_cast<oroDeviceptr>(d_s_b));
  if (d_noise_a) (void)oroFree(reinterpret_cast<oroDeviceptr>(d_noise_a));
  if (d_noise_b) (void)oroFree(reinterpret_cast<oroDeviceptr>(d_noise_b));
  if (d_partial) (void)oroFree(reinterpret_cast<oroDeviceptr>(d_partial));
  if (d_jackpot) (void)oroFree(reinterpret_cast<oroDeviceptr>(d_jackpot));
  (void)oroCtxDestroy(ctx);

  return ok ? 0 : 1;
}

#endif

} // namespace

std::array<uint8_t, kIncompleteBlockHeaderBytes> IncompleteBlockHeader::to_bytes() const
{
  std::vector<uint8_t> out;
  out.reserve(kIncompleteBlockHeaderBytes);

  append_u32_le(out, version);
  out.insert(out.end(), prev_block.rbegin(), prev_block.rend());
  out.insert(out.end(), merkle_root.rbegin(), merkle_root.rend());
  append_u32_le(out, timestamp);
  append_u32_le(out, nbits);

  std::array<uint8_t, kIncompleteBlockHeaderBytes> bytes{};
  std::copy_n(out.begin(), bytes.size(), bytes.begin());
  return bytes;
}

std::array<uint8_t, 2 * kPatternDims> PeriodicPattern::to_bytes() const
{
  std::array<uint8_t, 2 * kPatternDims> bytes{};
  uint32_t min_stride = 1;

  for (size_t i = 0; i < shape.size(); ++i)
  {
    const auto [stride, length] = shape[i];
    const uint32_t factor = stride / min_stride;
    bytes[2 * i] = static_cast<uint8_t>(factor - 1);
    bytes[2 * i + 1] = static_cast<uint8_t>(length - 1);
    min_stride = stride * length;
  }

  return bytes;
}

std::vector<uint32_t> PeriodicPattern::to_list() const
{
  std::vector<uint32_t> result{0};
  for (const auto& [stride, length] : shape)
  {
    std::vector<uint32_t> next;
    next.reserve(result.size() * length);
    for (uint32_t i = 0; i < length; ++i)
    {
      for (uint32_t value : result)
      {
        next.push_back(value + i * stride);
      }
    }
    result = std::move(next);
  }
  return result;
}

uint32_t PeriodicPattern::period() const
{
  const auto [stride, length] = shape.back();
  return stride * length;
}

uint32_t PeriodicPattern::size() const
{
  uint32_t total = 1;
  for (const auto& [_, length] : shape)
  {
    total *= length;
  }
  return total;
}

std::array<uint8_t, kMiningConfigBytes> MiningConfiguration::to_bytes() const
{
  std::vector<uint8_t> out;
  out.reserve(kMiningConfigBytes);

  append_u32_le(out, common_dim);
  append_u16_le(out, rank);
  append_u16_le(out, mma_type);

  const auto rows = rows_pattern.to_bytes();
  const auto cols = cols_pattern.to_bytes();
  out.insert(out.end(), rows.begin(), rows.end());
  out.insert(out.end(), cols.begin(), cols.end());
  out.insert(out.end(), reserved.begin(), reserved.end());

  std::array<uint8_t, kMiningConfigBytes> bytes{};
  std::copy_n(out.begin(), bytes.size(), bytes.begin());
  return bytes;
}

uint32_t MiningConfiguration::dot_product_length() const
{
  return common_dim - (common_dim % rank);
}

Hash256 blake3_digest(const uint8_t* data, size_t len)
{
  Hash256 out{};
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, data, len);
  blake3_hasher_finalize(&hasher, out.data(), out.size());
  return out;
}

Hash256 blake3_keyed_digest(const uint8_t* data, size_t len, const Hash256& key)
{
  Hash256 out{};
  blake3_hasher hasher;
  blake3_hasher_init_keyed(&hasher, key.data());
  blake3_hasher_update(&hasher, data, len);
  blake3_hasher_finalize(&hasher, out.data(), out.size());
  return out;
}

Hash256 compute_job_key(const IncompleteBlockHeader& header, const MiningConfiguration& config)
{
  const auto header_bytes = header.to_bytes();
  const auto config_bytes = config.to_bytes();

  std::array<uint8_t, kIncompleteBlockHeaderBytes + kMiningConfigBytes> input{};
  std::copy(header_bytes.begin(), header_bytes.end(), input.begin());
  std::copy(config_bytes.begin(), config_bytes.end(), input.begin() + header_bytes.size());
  return blake3_digest(input.data(), input.size());
}

std::pair<Hash256, Hash256> compute_commitment_hash(const Hash256& job_key, const Hash256& hash_a, const Hash256& hash_b)
{
  std::array<uint8_t, 64> b_seed_input{};
  std::copy(job_key.begin(), job_key.end(), b_seed_input.begin());
  std::copy(hash_b.begin(), hash_b.end(), b_seed_input.begin() + job_key.size());
  const Hash256 b_noise_seed = blake3_digest(b_seed_input.data(), b_seed_input.size());

  std::array<uint8_t, 64> a_seed_input{};
  std::copy(b_noise_seed.begin(), b_noise_seed.end(), a_seed_input.begin());
  std::copy(hash_a.begin(), hash_a.end(), a_seed_input.begin() + b_noise_seed.size());
  const Hash256 a_noise_seed = blake3_digest(a_seed_input.data(), a_seed_input.size());

  return {b_noise_seed, a_noise_seed};
}

Hash256 compute_jackpot_hash(const std::array<uint32_t, 16>& jackpot, const Hash256& a_noise_seed)
{
  std::array<uint8_t, 64> msg{};
  for (size_t i = 0; i < msg.size(); ++i)
  {
    msg[i] = static_cast<uint8_t>((jackpot[i / 4] >> (8 * (i % 4))) & 0xff);
  }
  return blake3_keyed_digest(msg.data(), msg.size(), a_noise_seed);
}

HarnessFixture default_fixture()
{
  HarnessFixture fixture{};
  fixture.name = "openpearl-python-api-default";
  fixture.m = 256;
  fixture.n = 128;
  fixture.header.version = 0;
  fixture.header.timestamp = 0x66666666;
  fixture.header.nbits = 0x1d2fffff;

  static constexpr uint8_t merkle[] = "0123456789abcdef0123456789abcdef";
  std::copy_n(merkle, fixture.header.merkle_root.size(), fixture.header.merkle_root.begin());

  fixture.config.common_dim = 1024;
  fixture.config.rank = 32;
  fixture.config.mma_type = kMmaInt7xInt7ToInt32;
  fixture.config.rows_pattern.shape = {{{8, 2}, {64, 2}, {128, 1}}};
  fixture.config.cols_pattern.shape = {{{1, 2}, {8, 2}, {32, 2}}};
  return fixture;
}

static bool run_fixture_contract_checks(const char* tag, const HarnessFixture& fixture)
{
  bool ok = true;

  ok &= check_u32_list(tag, "rows_pattern", fixture.config.rows_pattern.to_list(), {0, 8, 64, 72});
  ok &= check_u32_list(tag, "cols_pattern", fixture.config.cols_pattern.to_list(), {0, 1, 8, 9, 32, 33, 40, 41});
  ok &= fixture.config.rows_pattern.size() == 4;
  ok &= fixture.config.cols_pattern.size() == 8;
  ok &= fixture.config.dot_product_length() == 1024;

  const auto header_bytes = fixture.header.to_bytes();
  const auto config_bytes = fixture.config.to_bytes();
  const auto job_key = compute_job_key(fixture.header, fixture.config);

  ok &= check_bytes(
    tag,
    "header_bytes",
    hex_bytes(header_bytes),
    "000000000000000000000000000000000000000000000000000000000000000000000000666564636261393837363534333231306665646362613938373635343332313066666666ffff2f1d");
  ok &= check_bytes(
    tag,
    "mining_config_bytes",
    hex_bytes(config_bytes),
    "00040000200000000701030100000001030101010000000000000000000000000000000000000000000000000000000000000000");

  TNN_LOG_INFO("%s fixture=%s m=%u n=%u k=%u rank=%u\n",
               tag,
               fixture.name,
               fixture.m,
               fixture.n,
               fixture.config.common_dim,
               fixture.config.rank);
  TNN_LOG_INFO("%s job_key=%s\n", tag, hex_bytes(job_key).c_str());

  return ok;
}

static bool run_openpearl_vector_checks(const char* tag, const HarnessFixture& fixture)
{
  constexpr const char* vector_name = "openpearl-python-api-default-chacha20-deadbeef";
  constexpr uint64_t attempts = 4;
  constexpr uint32_t t_rows = 16;
  constexpr uint32_t t_cols = 16;

  const std::vector<uint32_t> a_row_indices = {16, 24, 80, 88};
  const std::vector<uint32_t> bt_row_indices = {16, 17, 24, 25, 48, 49, 56, 57};

  const Hash256 hash_a = hash_from_hex("88c7adb4f1a8d0144fd69ab6a827eba27352a64d506f313c6ceb238a2d50730e");
  const Hash256 hash_b = hash_from_hex("2e3f1797a80ab8028a9524604caad12c036059fcfaceaa601fd95993c5db9034");
  const Hash256 expected_b_noise_seed = hash_from_hex("c20cf3fd6d5f5e8ccab11a63e4832fbe9db36c7bc8bb60c1faadd40058c48791");
  const Hash256 expected_a_noise_seed = hash_from_hex("1134748ecce2eadc7ed91a91a5f3cf9c9c806205afa0399a3d04fd43d64a2323");
  const Hash256 expected_hash_jackpot = hash_from_hex("87c814042be240802805a62bb7438d6129837d45edc60d8b256ea6f892ab0900");
  const std::array<uint32_t, 16> expected_jackpot = {
    0x07ac6dcd, 0x1e82a302, 0xdb625818, 0x20727c2f,
    0xd273c89c, 0x2f926839, 0x1a9fda3d, 0xe0a8ca8b,
    0xd7c521fd, 0x0389ee21, 0xdfe51ce8, 0xec6827e0,
    0xe03ced35, 0x24ae4dba, 0x0870e804, 0xe5fdb03b,
  };

  bool ok = true;
  const Hash256 job_key = compute_job_key(fixture.header, fixture.config);
  const auto [b_noise_seed, a_noise_seed] = compute_commitment_hash(job_key, hash_a, hash_b);
  const I32Matrix s_a = matrix_from_hex_rows(vectors::kOpenPearlSAHex, fixture.config.common_dim);
  const I32Matrix s_b = matrix_from_hex_rows(vectors::kOpenPearlSBHex, fixture.config.common_dim);
  const I32Matrix noise_a = matrix_from_hex_rows(vectors::kOpenPearlNoiseAHex, fixture.config.common_dim);
  const I32Matrix noise_b = matrix_from_hex_rows(vectors::kOpenPearlNoiseBHex, fixture.config.common_dim);
  const std::array<uint32_t, 16> jackpot = compute_jackpot_words(
    s_a,
    s_b,
    noise_a,
    noise_b,
    fixture.config.common_dim,
    fixture.config.rank);
  const Hash256 hash_jackpot = compute_jackpot_hash(jackpot, a_noise_seed);

  ok &= check_u32_list(tag, "vector.a_row_indices", a_row_indices, {16, 24, 80, 88});
  ok &= check_u32_list(tag, "vector.bt_row_indices", bt_row_indices, {16, 17, 24, 25, 48, 49, 56, 57});
  ok &= check_bytes(tag, "vector.b_noise_seed", hex_bytes(b_noise_seed), hex_bytes(expected_b_noise_seed).c_str());
  ok &= check_bytes(tag, "vector.a_noise_seed", hex_bytes(a_noise_seed), hex_bytes(expected_a_noise_seed).c_str());
  ok &= check_jackpot_words(tag, jackpot, expected_jackpot);
  ok &= check_bytes(tag, "vector.hash_jackpot", hex_bytes(hash_jackpot), hex_bytes(expected_hash_jackpot).c_str());

  TNN_LOG_INFO("%s vector=%s attempts=%llu t_rows=%u t_cols=%u\n",
               tag,
               vector_name,
               static_cast<unsigned long long>(attempts),
               t_rows,
               t_cols);
  TNN_LOG_INFO("%s hash_a=%s\n", tag, hex_bytes(hash_a).c_str());
  TNN_LOG_INFO("%s hash_b=%s\n", tag, hex_bytes(hash_b).c_str());

#ifdef TNN_HIP
  ok &= run_noise_gpu_checks(tag, fixture, a_noise_seed, b_noise_seed);
  ok &= run_jackpot_gpu_check(
    tag,
    s_a, s_b, noise_a, noise_b,
    static_cast<int>(fixture.config.common_dim),
    static_cast<int>(fixture.config.rank),
    expected_jackpot);
#endif

  return ok;
}

} // namespace tnn::pearl

int test_pearl_hip()
{
  constexpr const char* tag = "[PEARL-HIP-TEST]";
  TNN_LOG_INFO_COLOR(BRIGHT_CYAN, "%s Pearl PoUW HIP harness scaffold\n", tag);
  TNN_LOG_INFO("%s proto_solo=%d algo=%d\n", tag, PROTO_PEARL_SOLO, ALGO_PEARL_POUW);
  const auto fixture = tnn::pearl::default_fixture();

  if (!tnn::pearl::run_fixture_contract_checks(tag, fixture))
  {
    TNN_LOG_ERROR("%s OpenPearl fixture contract check failed\n", tag);
    return 1;
  }

  if (!tnn::pearl::run_openpearl_vector_checks(tag, fixture))
  {
    TNN_LOG_ERROR("%s OpenPearl vector check failed\n", tag);
    return 1;
  }

  TNN_LOG_INFO("%s TODO: add GPU dense-sparse expansion and OpenPearl PlainProof vectors\n", tag);
  return 0;
}

int bench_pearl_hip()
{
#ifdef TNN_HIP
  constexpr const char* tag = "[PEARL-HIP-BENCH]";
  return tnn::pearl::run_pearl_benchmark(tag);
#else
  TNN_LOG_ERROR("[PEARL-HIP-BENCH] ERROR: TNN_HIP is not enabled\n");
  return 1;
#endif
}
