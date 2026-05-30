#include "test_pearl_hip.h"

#include <tnn_hip/crypto/pearl/pearl_pouw_defs.h>

#include <BLAKE3/c/blake3.h>

#include <algo_definitions.h>
#include <tnn_log.hpp>

#ifdef TNN_HIP
#include <tnn_hip/common/gpu_algo.hpp>
#include <tnn_hip/common/gpu_rtc.hpp>
#include "iris_embedded_headers.hpp"
#include "pearl_embedded_headers.hpp"
#include "pearl_gemm_simple.hip.hpp"
#include "pearl_fused_noise_gemm.hip.hpp"
#include "pearl_sparse_noise_gen.hip.hpp"
#include "rocwmma_headers.hip.hpp"
#include "tnn_hip_common_embedded.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace tnn::pearl {
namespace {

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

static inline int32_t clampi8_host_jkp(int32_t v)
{
  return v < -128 ? -128 : (v > 127 ? 127 : v); // no-op with default signal range, but here for semantic indication
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

  std::vector<std::vector<int64_t>> jackpot(h, std::vector<int64_t>(w, 0));
  std::array<uint32_t, jackpot_size> jackpot_msg{};

  for (size_t ll = rank; ll <= k; ll += rank)
  {
    for (size_t u = 0; u < h; ++u)
    {
      for (size_t v = 0; v < w; ++v)
      {
        for (size_t l = ll - rank; l < ll; ++l)
        {
          const int32_t a = clampi8_host_jkp(s_a[u][l] + noise_a[u][l]);
          const int32_t b = clampi8_host_jkp(s_b[v][l] + noise_b[v][l]);

          jackpot[u][v] += static_cast<int64_t>(a) * static_cast<int64_t>(b);
        }
      }
    }

    uint32_t xored_tile = 0;
    for (const auto& row : jackpot)
    {
      for (int64_t value : row)
      {
        xored_tile ^= static_cast<uint32_t>(value);
      }
    }

    const size_t tid = (ll / rank - 1) % jackpot_size;
    jackpot_msg[tid] =
      ((jackpot_msg[tid] << lrot_per_tile) |
       (jackpot_msg[tid] >> (32 - lrot_per_tile))) ^
      xored_tile;
  }

  return jackpot_msg;
}

std::vector<uint32_t> compute_per_tile_xor(
  const I32Matrix& s_a,
  const I32Matrix& s_b,
  const I32Matrix& noise_a,
  const I32Matrix& noise_b,
  size_t k,
  size_t rank)
{
  const size_t h = s_a.size();
  const size_t w = s_b.size();
  const size_t num_tiles = k / rank;
  std::vector<uint32_t> tile_xor(num_tiles, 0);

  for (size_t tile = 0; tile < num_tiles; ++tile)
  {
    const size_t start_l = tile * rank;
    const size_t end_l = start_l + rank;
    int32_t xored = 0;
    for (size_t u = 0; u < h; ++u)
    {
      for (size_t v = 0; v < w; ++v)
      {
        int32_t sum = 0;
        for (size_t l = start_l; l < end_l; ++l)
        {
          sum += (s_a[u][l] + noise_a[u][l]) * (s_b[v][l] + noise_b[v][l]);
        }
        xored ^= sum;
      }
    }
    tile_xor[tile] = static_cast<uint32_t>(xored);
  }
  return tile_xor;
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

int8_t pearl_dense_noise_byte(uint8_t raw)
{
  constexpr int noise_abs_max = 128;
  constexpr int noise_range = 64;
  const int32_t signed_raw = static_cast<int32_t>(static_cast<int8_t>(raw));
  return static_cast<int8_t>(((signed_raw + noise_abs_max) % noise_range) - (noise_range / 2));
}

inline int8_t clampi8(int32_t v)
{
  if (v < -128) return -128;
  if (v > 127) return 127;
  return static_cast<int8_t>(v);
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

std::vector<int32_t> compute_host_gemm_ref(
  const std::vector<int8_t>& ApEA,
  const std::vector<int8_t>& BpEB,
  int h, int w, int k)
{
  std::vector<int32_t> C(static_cast<size_t>(h) * w, 0);
  for (int i = 0; i < h; ++i)
  {
    for (int j = 0; j < w; ++j)
    {
      int32_t sum = 0;
      for (int l = 0; l < k; ++l)
      {
        sum += static_cast<int32_t>(ApEA[static_cast<size_t>(i) * k + l]) *
               static_cast<int32_t>(BpEB[static_cast<size_t>(j) * k + l]);
      }
      C[static_cast<size_t>(i) * w + j] = sum;
    }
  }
  return C;
}

std::pair<std::vector<int8_t>, std::vector<int8_t>>
compute_noised_matrices(
  const I32Matrix& s_a, const I32Matrix& s_b,
  const I32Matrix& noise_a, const I32Matrix& noise_b)
{
  int h = static_cast<int>(s_a.size());
  int w = static_cast<int>(s_b.size());
  int k = static_cast<int>(s_a[0].size());

  std::vector<int8_t> ApEA(static_cast<size_t>(h) * k);
  std::vector<int8_t> BpEB(static_cast<size_t>(w) * k);

  for (int i = 0; i < h; ++i)
    for (int l = 0; l < k; ++l)
      ApEA[static_cast<size_t>(i) * k + l] =
        clampi8(s_a[i][l] + noise_a[i][l]);

  for (int j = 0; j < w; ++j)
    for (int l = 0; l < k; ++l)
      BpEB[static_cast<size_t>(j) * k + l] =
        clampi8(s_b[j][l] + noise_b[j][l]);

  return {std::move(ApEA), std::move(BpEB)};
}

struct SparseNoisePairs
{
  std::vector<int> first_idx;
  std::vector<int> second_idx;
};

SparseNoisePairs extract_sparse_pairs(
  const std::vector<int8_t>& sparse_noise,
  size_t num_rows,
  size_t cols)
{
  SparseNoisePairs pairs;
  pairs.first_idx.resize(num_rows);
  pairs.second_idx.resize(num_rows);
  for (size_t row = 0; row < num_rows; ++row)
  {
    bool found_first = false;
    for (size_t c = 0; c < cols; ++c)
    {
      if (sparse_noise[row * cols + c] == 1)
      {
        if (!found_first)
        {
          pairs.first_idx[row] = static_cast<int>(c);
          found_first = true;
        }
      }
      else if (sparse_noise[row * cols + c] == -1)
      {
        pairs.second_idx[row] = static_cast<int>(c);
      }
    }
  }
  return pairs;
}

I32Matrix compute_noise_from_factors_dense_times_sparse_T(
  const std::vector<int8_t>& dense,
  size_t dense_rows,
  const SparseNoisePairs& sparse_pairs,
  size_t sparse_rows,
  size_t R)
{
  I32Matrix noise(dense_rows, std::vector<int32_t>(sparse_rows, 0));
  for (size_t i = 0; i < dense_rows; ++i)
  {
    for (size_t l = 0; l < sparse_rows; ++l)
    {
      int32_t val = static_cast<int32_t>(dense[i * R + sparse_pairs.first_idx[l]]) -
                    static_cast<int32_t>(dense[i * R + sparse_pairs.second_idx[l]]);
      noise[i][l] = val;
    }
  }
  return noise;
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

bool pearl_register_rtc_headers()
{
  auto& compiler = RTCCompiler::instance();
  auto rtc_headers = build_rtc_headers(
    hip_embedded::PEARL_HEADERS,
    hip_embedded::IRIS_HEADERS,
    hip_embedded::COMMON_HEADERS,
    hip_embedded::ROCWMMA_HEADERS);
  for (const auto& hdr : rtc_headers)
  {
    const std::string include_name(hdr.name);
    const std::string source(hdr.source);
    compiler.add_header_source(include_name, source);

    const std::string basename = rtc_include_basename(include_name);
    if (basename != include_name)
      compiler.add_header_source(basename, source);

    {
      auto parent = include_name;
      auto slash  = parent.find('/');
      while (slash != std::string::npos)
      {
        parent = parent.substr(slash + 1);
        compiler.add_header_source(parent, source);
        slash = parent.find('/');
      }
    }
  }
  return true;
}

#endif

} // namespace

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

} // namespace tnn::pearl

static int env_or_default(const char* name, int fallback)
{
  const char* val = std::getenv(name);
  if (!val) return fallback;
  char* end = nullptr;
  long parsed = std::strtol(val, &end, 10);
  if (end == val || *end != '\0') return fallback;
  return static_cast<int>(parsed);
}

namespace tnn::pearl {

namespace {
constexpr uint32_t TBLOCK_X       = 64u;
constexpr uint32_t TBLOCK_Y       = 2u;
constexpr uint32_t WARP_SIZE      = 32u;
constexpr uint32_t ROCWMMA_M      = 16u;
constexpr uint32_t ROCWMMA_N      = 16u;
constexpr uint32_t BLOCKS_M       = 4u;
constexpr uint32_t BLOCKS_N       = 2u;
constexpr uint32_t WARP_TILE_M    = BLOCKS_M * ROCWMMA_M;
constexpr uint32_t WARP_TILE_N    = BLOCKS_N * ROCWMMA_N;
constexpr uint32_t WARPS_M        = TBLOCK_X / WARP_SIZE;
constexpr uint32_t WARPS_N        = TBLOCK_Y;
constexpr uint32_t MACRO_TILE_M   = WARPS_M * WARP_TILE_M;
constexpr uint32_t MACRO_TILE_N   = WARPS_N * WARP_TILE_N;
constexpr uint32_t MACRO_TILE_K   = 16u;
} // anonymous namespace

static bool test_pearl_noised_gemm(
  const char* tag,
  const RTCCompiler::CompiledKernel& compiled)
{
  setvbuf(stdout, NULL, _IONBF, 0);
  TNN_LOG_INFO("%s [NG] starting noised GEMM test\n", tag);

  const HarnessFixture fixture = default_fixture();
  TNN_LOG_INFO("%s [NG] fixture: m=%u n=%u common_dim=%u rank=%u\n", tag,
               fixture.m, fixture.n, fixture.config.common_dim, fixture.config.rank);
  TNN_LOG_INFO("%s [NG] ROWS_PATTERN cols_pattern OK\n", tag);

  const int h = static_cast<int>(fixture.m);
  const int w = static_cast<int>(fixture.n);
  const int k = static_cast<int>(fixture.config.common_dim);
  const int R = static_cast<int>(fixture.config.rank);

  TNN_LOG_INFO("%s [NG] generating deterministic base s_a(%dx%d) and s_b(%dx%d)...\n", tag, h, k, w, k);

  I32Matrix s_a(h, std::vector<int32_t>(k));
  I32Matrix s_b(w, std::vector<int32_t>(k));
  // s_a, s_b elements in [-64, 64] to match Rust SIGNAL_MIN/SIGNAL_MAX
  for (int i = 0; i < h; ++i)
    for (int l = 0; l < k; ++l)
      s_a[i][l] = static_cast<int32_t>(((i * k + l) * 0x9e3779b9u) % 129 - 64);
  for (int j = 0; j < w; ++j)
    for (int l = 0; l < k; ++l)
      s_b[j][l] = static_cast<int32_t>(((j * k + l) * 0x9e3779b9u + 0x55) % 129 - 64);

  TNN_LOG_INFO("%s [NG] computing hashes...\n", tag);
  const Hash256 hash_a = hash_from_hex("88c7adb4f1a8d0144fd69ab6a827eba27352a64d506f313c6ceb238a2d50730e");
  const Hash256 hash_b = hash_from_hex("2e3f1797a80ab8028a9524604caad12c036059fcfaceaa601fd95993c5db9034");
  const Hash256 job_key = compute_job_key(fixture.header, fixture.config);
  const auto [b_noise_seed, a_noise_seed] = compute_commitment_hash(job_key, hash_a, hash_b);
  TNN_LOG_INFO("%s [NG] hashes computed\n", tag);

  std::array<uint8_t, 32> seed_a_label = {};
  std::copy_n("A_tensor", 8, seed_a_label.begin());
  std::array<uint8_t, 32> seed_b_label = {};
  std::copy_n("B_tensor", 8, seed_b_label.begin());

  TNN_LOG_INFO("%s [NG] generating dense eal(%dx%d)...\n", tag, h, R);
  const std::vector<int8_t> eal = pearl_dense_noise_reference(h, R, a_noise_seed, seed_a_label);
  TNN_LOG_INFO("%s [NG] eal size=%zu\n", tag, eal.size());

  TNN_LOG_INFO("%s [NG] generating sparse ear(%dx%d)...\n", tag, k, R);
  const std::vector<int8_t> ear = pearl_sparse_noise_reference(k, R, a_noise_seed, seed_a_label);
  TNN_LOG_INFO("%s [NG] ear size=%zu\n", tag, ear.size());

  TNN_LOG_INFO("%s [NG] generating sparse ebl(%dx%d)...\n", tag, k, R);
  const std::vector<int8_t> ebl = pearl_sparse_noise_reference(k, R, b_noise_seed, seed_b_label);
  TNN_LOG_INFO("%s [NG] ebl size=%zu\n", tag, ebl.size());

  TNN_LOG_INFO("%s [NG] generating dense ebr(%dx%d)...\n", tag, w, R);
  const std::vector<int8_t> ebr = pearl_dense_noise_reference(w, R, b_noise_seed, seed_b_label);
  TNN_LOG_INFO("%s [NG] ebr size=%zu\n", tag, ebr.size());

  TNN_LOG_INFO("%s [NG] extracting sparse pairs...\n", tag);
  const SparseNoisePairs ear_pairs = extract_sparse_pairs(ear, k, R);
  TNN_LOG_INFO("%s [NG] ear_pairs extracted: first_idx size=%zu\n", tag, ear_pairs.first_idx.size());
  const SparseNoisePairs ebl_pairs = extract_sparse_pairs(ebl, k, R);
  TNN_LOG_INFO("%s [NG] ebl_pairs extracted: first_idx size=%zu\n", tag, ebl_pairs.first_idx.size());

  TNN_LOG_INFO("%s [NG] composing noise_a (%dx%d)...\n", tag, h, k);
  const I32Matrix noise_a = compute_noise_from_factors_dense_times_sparse_T(eal, h, ear_pairs, k, R);
  TNN_LOG_INFO("%s [NG] noise_a composed: %zu x %zu\n", tag, noise_a.size(), noise_a.empty() ? 0 : noise_a[0].size());

  TNN_LOG_INFO("%s [NG] composing noise_b (%dx%d)...\n", tag, w, k);
  const I32Matrix noise_b = compute_noise_from_factors_dense_times_sparse_T(ebr, w, ebl_pairs, k, R);
  TNN_LOG_INFO("%s [NG] noise_b composed: %zu x %zu\n", tag, noise_b.size(), noise_b.empty() ? 0 : noise_b[0].size());

  TNN_LOG_INFO("%s [NG] computing noised matrices ApEA, BpEB...\n", tag);
  const auto [ApEA, BpEB] = compute_noised_matrices(s_a, s_b, noise_a, noise_b);
  TNN_LOG_INFO("%s [NG] ApEA size=%zu, BpEB size=%zu\n", tag, ApEA.size(), BpEB.size());

  TNN_LOG_INFO("%s [NG] computing host ref GEMM (%dx%dx%d)...\n", tag, h, w, k);
  const std::vector<int32_t> C_ref = compute_host_gemm_ref(ApEA, BpEB, h, w, k);
  TNN_LOG_INFO("%s [NG] C_ref size=%zu, first few: %d %d %d %d\n", tag,
               C_ref.size(),
               C_ref.size() > 0 ? C_ref[0] : 0,
               C_ref.size() > 1 ? C_ref[1] : 0,
               C_ref.size() > 2 ? C_ref[2] : 0,
               C_ref.size() > 3 ? C_ref[3] : 0);

  TNN_LOG_INFO("%s [NG] noised GEMM ref computed: h=%d w=%d k=%d R=%d\n", tag, h, w, k, R);

  TNN_LOG_INFO("%s [NG] preparing GPU buffers...\n", tag);
  const std::size_t a_elems = static_cast<std::size_t>(k) * h;
  const std::size_t b_elems = static_cast<std::size_t>(k) * w;
  const std::size_t c_elems = static_cast<std::size_t>(h) * w;
  TNN_LOG_INFO("%s [NG] a_elems=%zu b_elems=%zu c_elems=%zu\n", tag, a_elems, b_elems, c_elems);

  std::vector<signed char> hA(a_elems);
  std::vector<signed char> hB(b_elems);
  std::vector<int32_t> hC(c_elems, 0);

  TNN_LOG_INFO("%s [NG] copying A row-major (%dx%d)...\n", tag, h, k);
  for (std::size_t i = 0; i < a_elems; ++i) hA[i] = ApEA[i];
  TNN_LOG_INFO("%s [NG] copying B row-major (%dx%d)...\n", tag, w, k);
  for (std::size_t i = 0; i < b_elems; ++i) hB[i] = BpEB[i];

  TNN_LOG_INFO("%s [NG] GPU oroMalloc...\n", tag);
  signed char* dA = nullptr;
  signed char* dB = nullptr;
  int32_t* dC = nullptr;

  bool ok = true;
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&dA), a_elems));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&dB), b_elems));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&dC), c_elems * sizeof(int32_t)));

  if (!ok)
  {
    TNN_LOG_ERROR("%s [NG] oroMalloc failed\n", tag);
    if (dA) (void)oroFree(dA);
    if (dB) (void)oroFree(dB);
    if (dC) (void)oroFree(dC);
    return false;
  }

  ok  = PEARL_GPU_CHECK(oroMemcpy(dA, hA.data(), a_elems, oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(dB, hB.data(), b_elems, oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemset(dC, 0, c_elems * sizeof(int32_t)));

  if (!ok)
  {
    TNN_LOG_ERROR("%s [NG] oroMemcpy failed\n", tag);
    (void)oroFree(dA);
    (void)oroFree(dB);
    (void)oroFree(dC);
    return false;
  }

  int M_gpu = h, N_gpu = w, K_gpu = k;
  const dim3 gridDim(
    (M_gpu + MACRO_TILE_M - 1) / MACRO_TILE_M,
    (N_gpu + MACRO_TILE_N - 1) / MACRO_TILE_N,
    1);
  const dim3 blockDim(TBLOCK_X, TBLOCK_Y, 1);

  constexpr uint32_t ldsWidth      = MACRO_TILE_K;
  constexpr uint32_t ldsHeightA    = MACRO_TILE_M;
  constexpr uint32_t ldsHeightB    = MACRO_TILE_N;
  constexpr uint32_t ldsHeight     = ldsHeightA + ldsHeightB;
  constexpr uint32_t sizeLds       = ldsHeight * ldsWidth;
  const int sharedMemBytes = static_cast<int>(2 * sizeLds * sizeof(signed char));

  int lda = K_gpu;
  int ldb = K_gpu;
  int ldd = N_gpu;

  void* kernel_args[] = {&M_gpu, &N_gpu, &K_gpu, &dA, &dB, &dC, &lda, &ldb, &ldd};

  TNN_LOG_INFO("%s [NG] launching rocWMMA kernel grid=(%u,%u) block=(%u,%u) shmem=%d\n",
               tag, gridDim.x, gridDim.y, blockDim.x, blockDim.y, sharedMemBytes);
  ok = PEARL_GPU_CHECK(oroModuleLaunchKernel(
    compiled.function,
    gridDim.x, gridDim.y, gridDim.z,
    blockDim.x, blockDim.y, blockDim.z,
    sharedMemBytes, nullptr, kernel_args, nullptr));

  ok &= PEARL_GPU_CHECK(oroDeviceSynchronize());

  if (!ok)
  {
    TNN_LOG_ERROR("%s [NG] Kernel launch or sync failed\n", tag);
    (void)oroFree(dA);
    (void)oroFree(dB);
    (void)oroFree(dC);
    return false;
  }

  TNN_LOG_INFO("%s [NG] kernel done, downloading result...\n", tag);

  ok = PEARL_GPU_CHECK(oroMemcpy(hC.data(), dC, c_elems * sizeof(int32_t), oroMemcpyDeviceToHost));
  if (!ok)
  {
    TNN_LOG_ERROR("%s [NG] oroMemcpy result failed\n", tag);
    (void)oroFree(dA);
    (void)oroFree(dB);
    (void)oroFree(dC);
    return false;
  }

  (void)oroFree(dA);
  (void)oroFree(dB);
  (void)oroFree(dC);

  TNN_LOG_INFO("%s [NG] comparing %d elements...\n", tag, h * w);
  int64_t max_abs_error = 0;
  int64_t mismatch_count = 0;
  int first_mismatch_i = -1, first_mismatch_j = -1;
  int32_t first_mismatch_ref = 0, first_mismatch_gpu = 0;

  for (int i = 0; i < h; ++i)
  {
    for (int j = 0; j < w; ++j)
    {
      const size_t idx = static_cast<size_t>(i) * w + j;
      const int64_t delta = static_cast<int64_t>(C_ref[idx]) - static_cast<int64_t>(hC[idx]);
      const int64_t abs_delta = (delta < 0) ? -delta : delta;
      if (abs_delta > 0)
      {
        ++mismatch_count;
        if (abs_delta > max_abs_error)
          max_abs_error = abs_delta;
        if (first_mismatch_i < 0)
        {
          first_mismatch_i = i;
          first_mismatch_j = j;
          first_mismatch_ref = C_ref[idx];
          first_mismatch_gpu = hC[idx];
        }
      }
    }
  }

  if (mismatch_count == 0)
  {
    TNN_LOG_INFO("%s noised GEMM: PASS (%d elements match exactly)\n", tag, h * w);
    return true;
  }

  TNN_LOG_ERROR("%s noised GEMM: FAIL\n", tag);
  TNN_LOG_ERROR("%s   total elements: %d\n", tag, h * w);
  TNN_LOG_ERROR("%s   mismatches: %ld\n", tag, static_cast<long>(mismatch_count));
  TNN_LOG_ERROR("%s   max_abs_error: %ld\n", tag, static_cast<long>(max_abs_error));
  TNN_LOG_ERROR("%s   first mismatch at (%d, %d): ref=%d gpu=%d\n",
                tag, first_mismatch_i, first_mismatch_j,
                 first_mismatch_ref, first_mismatch_gpu);
  return false;
}

static bool test_pearl_sparse_noise(
  const char* tag,
  const std::vector<std::string>& compile_opts)
{
  setvbuf(stdout, NULL, _IONBF, 0);
  TNN_LOG_INFO("%s [SPN] starting sparse noise gen test\n", tag);

  const HarnessFixture fixture = default_fixture();
  const int k = static_cast<int>(fixture.config.common_dim);
  const int R = static_cast<int>(fixture.config.rank);

  const Hash256 hash_a = hash_from_hex("88c7adb4f1a8d0144fd69ab6a827eba27352a64d506f313c6ceb238a2d50730e");
  const Hash256 hash_b = hash_from_hex("2e3f1797a80ab8028a9524604caad12c036059fcfaceaa601fd95993c5db9034");
  const Hash256 job_key = compute_job_key(fixture.header, fixture.config);
  const auto [b_noise_seed, a_noise_seed] = compute_commitment_hash(job_key, hash_a, hash_b);

  std::array<uint8_t, 32> seed_a_label = {};
  std::copy_n("A_tensor", 8, seed_a_label.begin());
  std::array<uint8_t, 32> seed_b_label = {};
  std::copy_n("B_tensor", 8, seed_b_label.begin());

  TNN_LOG_INFO("%s [SPN] compiling sparse noise gen kernel via RTC...\n", tag);
  auto& compiler = RTCCompiler::instance();
  const std::string spn_source(
    hip_pearl_sparse_noise_gen_source::SRC_TNN_HIP_CRYPTO_PEARL_PEARL_SPARSE_NOISE_GEN_HIP_SOURCE);
  RTCCompiler::CompiledKernel spn_compiled{};
  try
  {
    spn_compiled = compiler.compile_from_source(
      spn_source,
      "pearl_sparse_noise_gen.hip",
      "pearl_sparse_noise_gen",
      compile_opts);
  }
  catch (const std::exception& e)
  {
    TNN_LOG_ERROR("%s [SPN] compile failed: %s\n", tag, e.what());
    return false;
  }
  TNN_LOG_INFO("%s [SPN] kernel compiled\n", tag);

  TNN_LOG_INFO("%s [SPN] computing CPU reference...\n", tag);
  const std::vector<int8_t> ear_ref = pearl_sparse_noise_reference(k, R, a_noise_seed, seed_a_label);
  const SparseNoisePairs ear_pairs_ref = extract_sparse_pairs(ear_ref, k, R);
  const std::vector<int8_t> ebl_ref = pearl_sparse_noise_reference(k, R, b_noise_seed, seed_b_label);
  const SparseNoisePairs ebl_pairs_ref = extract_sparse_pairs(ebl_ref, k, R);

  TNN_LOG_INFO("%s [SPN] allocating GPU buffers...\n", tag);
  int32_t* d_ear_first = nullptr;
  int32_t* d_ear_second = nullptr;
  int32_t* d_ebl_first = nullptr;
  int32_t* d_ebl_second = nullptr;
  uint8_t* d_key_a = nullptr;
  uint8_t* d_key_b = nullptr;
  uint8_t* d_seed_a = nullptr;
  uint8_t* d_seed_b = nullptr;

  bool ok = true;
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_ear_first),  k * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_ear_second), k * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_ebl_first),  k * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_ebl_second), k * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_key_a),   a_noise_seed.size()));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_key_b),   b_noise_seed.size()));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_seed_a),  seed_a_label.size()));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_seed_b),  seed_b_label.size()));

  if (!ok)
  {
    TNN_LOG_ERROR("%s [SPN] oroMalloc failed\n", tag);
    (void)oroFree(d_ear_first);  (void)oroFree(d_ear_second);
    (void)oroFree(d_ebl_first);  (void)oroFree(d_ebl_second);
    (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
    (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
    return false;
  }

  ok = PEARL_GPU_CHECK(oroMemcpy(d_key_a,  a_noise_seed.data(), a_noise_seed.size(), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_key_b,  b_noise_seed.data(), b_noise_seed.size(), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_seed_a, seed_a_label.data(), seed_a_label.size(), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_seed_b, seed_b_label.data(), seed_b_label.size(), oroMemcpyHostToDevice));

  if (!ok)
  {
    TNN_LOG_ERROR("%s [SPN] oroMemcpy failed\n", tag);
    (void)oroFree(d_ear_first);  (void)oroFree(d_ear_second);
    (void)oroFree(d_ebl_first);  (void)oroFree(d_ebl_second);
    (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
    (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
    return false;
  }

  const int K_gpu = k;
  const int R_gpu = R;
  const dim3 gridDim((K_gpu + 255) / 256, 1, 1);
  const dim3 blockDim(256, 1, 1);

  const uint32_t k_gpu = static_cast<uint32_t>(k);
  const uint32_t r_gpu = static_cast<uint32_t>(R);

  TNN_LOG_INFO("%s [SPN] launching ear sparse noise kernel...\n", tag);
  void* ear_args[] = {
    const_cast<uint32_t*>(&k_gpu),
    const_cast<uint32_t*>(&r_gpu),
    &d_key_a, &d_seed_a,
    &d_ear_first, &d_ear_second
  };
  ok = PEARL_GPU_CHECK(oroModuleLaunchKernel(
    spn_compiled.function,
    gridDim.x, gridDim.y, gridDim.z,
    blockDim.x, blockDim.y, blockDim.z,
    0, nullptr, ear_args, nullptr));
  ok &= PEARL_GPU_CHECK(oroDeviceSynchronize());

  if (!ok)
  {
    TNN_LOG_ERROR("%s [SPN] ear kernel failed\n", tag);
    (void)oroFree(d_ear_first);  (void)oroFree(d_ear_second);
    (void)oroFree(d_ebl_first);  (void)oroFree(d_ebl_second);
    (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
    (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
    return false;
  }

  TNN_LOG_INFO("%s [SPN] launching ebl sparse noise kernel...\n", tag);
  void* ebl_args[] = {
    const_cast<uint32_t*>(&k_gpu),
    const_cast<uint32_t*>(&r_gpu),
    &d_key_b, &d_seed_b,
    &d_ebl_first, &d_ebl_second
  };
  ok = PEARL_GPU_CHECK(oroModuleLaunchKernel(
    spn_compiled.function,
    gridDim.x, gridDim.y, gridDim.z,
    blockDim.x, blockDim.y, blockDim.z,
    0, nullptr, ebl_args, nullptr));
  ok &= PEARL_GPU_CHECK(oroDeviceSynchronize());

  if (!ok)
  {
    TNN_LOG_ERROR("%s [SPN] ebl kernel failed\n", tag);
    (void)oroFree(d_ear_first);  (void)oroFree(d_ear_second);
    (void)oroFree(d_ebl_first);  (void)oroFree(d_ebl_second);
    (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
    (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
    return false;
  }

  TNN_LOG_INFO("%s [SPN] downloading results...\n", tag);
  std::vector<int32_t> h_ear_first(k), h_ear_second(k);
  std::vector<int32_t> h_ebl_first(k), h_ebl_second(k);
  ok = PEARL_GPU_CHECK(oroMemcpy(h_ear_first.data(),  d_ear_first,  k * sizeof(int32_t), oroMemcpyDeviceToHost));
  ok &= PEARL_GPU_CHECK(oroMemcpy(h_ear_second.data(), d_ear_second, k * sizeof(int32_t), oroMemcpyDeviceToHost));
  ok &= PEARL_GPU_CHECK(oroMemcpy(h_ebl_first.data(),  d_ebl_first,  k * sizeof(int32_t), oroMemcpyDeviceToHost));
  ok &= PEARL_GPU_CHECK(oroMemcpy(h_ebl_second.data(), d_ebl_second, k * sizeof(int32_t), oroMemcpyDeviceToHost));

  (void)oroFree(d_ear_first);  (void)oroFree(d_ear_second);
  (void)oroFree(d_ebl_first);  (void)oroFree(d_ebl_second);
  (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
  (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);

  if (!ok)
  {
    TNN_LOG_ERROR("%s [SPN] download failed\n", tag);
    return false;
  }

  TNN_LOG_INFO("%s [SPN] comparing ear results (%d rows)...\n", tag, k);
  int64_t ear_mismatch = 0;
  for (int i = 0; i < k; ++i)
  {
    if (h_ear_first[i] != static_cast<int32_t>(ear_pairs_ref.first_idx[i]))
      ++ear_mismatch;
    if (h_ear_second[i] != static_cast<int32_t>(ear_pairs_ref.second_idx[i]))
      ++ear_mismatch;
  }

  TNN_LOG_INFO("%s [SPN] comparing ebl results (%d rows)...\n", tag, k);
  int64_t ebl_mismatch = 0;
  for (int i = 0; i < k; ++i)
  {
    if (h_ebl_first[i] != static_cast<int32_t>(ebl_pairs_ref.first_idx[i]))
      ++ebl_mismatch;
    if (h_ebl_second[i] != static_cast<int32_t>(ebl_pairs_ref.second_idx[i]))
      ++ebl_mismatch;
  }

  if (ear_mismatch == 0 && ebl_mismatch == 0)
  {
    TNN_LOG_INFO("%s sparse noise gen: PASS (all %d x8 pairs match exactly)\n", tag, k);
    return true;
  }

  TNN_LOG_ERROR("%s sparse noise gen: FAIL\n", tag);
  TNN_LOG_ERROR("%s   ear mismatches: %ld\n", tag, static_cast<long>(ear_mismatch));
  TNN_LOG_ERROR("%s   ebl mismatches: %ld\n", tag, static_cast<long>(ebl_mismatch));
  return false;
}

static bool test_pearl_fused_noise_gemm(
  const char* tag,
  const std::vector<std::string>& compile_opts)
{
  setvbuf(stdout, NULL, _IONBF, 0);
  TNN_LOG_INFO("%s [FNG] starting fused noise GEMM test\n", tag);

  const HarnessFixture fixture = default_fixture();
  TNN_LOG_INFO("%s [FNG] fixture: m=%u n=%u common_dim=%u rank=%u\n", tag,
               fixture.m, fixture.n, fixture.config.common_dim, fixture.config.rank);

  const int h = static_cast<int>(fixture.m);
  const int w = static_cast<int>(fixture.n);
  const int k = static_cast<int>(fixture.config.common_dim);
  const int R = static_cast<int>(fixture.config.rank);

  TNN_LOG_INFO("%s [FNG] generating deterministic base s_a(%dx%d) and s_b(%dx%d)...\n", tag, h, k, w, k);

  I32Matrix s_a(h, std::vector<int32_t>(k));
  I32Matrix s_b(w, std::vector<int32_t>(k));
  // s_a, s_b elements in [-64, 64] to match Rust SIGNAL_MIN/SIGNAL_MAX
  for (int i = 0; i < h; ++i)
    for (int l = 0; l < k; ++l)
      s_a[i][l] = static_cast<int32_t>(((i * k + l) * 0x9e3779b9u) % 129 - 64);
  for (int j = 0; j < w; ++j)
    for (int l = 0; l < k; ++l)
      s_b[j][l] = static_cast<int32_t>(((j * k + l) * 0x9e3779b9u + 0x55) % 129 - 64);

  TNN_LOG_INFO("%s [FNG] computing hashes...\n", tag);
  const Hash256 hash_a = hash_from_hex("88c7adb4f1a8d0144fd69ab6a827eba27352a64d506f313c6ceb238a2d50730e");
  const Hash256 hash_b = hash_from_hex("2e3f1797a80ab8028a9524604caad12c036059fcfaceaa601fd95993c5db9034");
  const Hash256 job_key = compute_job_key(fixture.header, fixture.config);
  const auto [b_noise_seed, a_noise_seed] = compute_commitment_hash(job_key, hash_a, hash_b);
  TNN_LOG_INFO("%s [FNG] hashes computed\n", tag);

  std::array<uint8_t, 32> seed_a_label = {};
  std::copy_n("A_tensor", 8, seed_a_label.begin());
  std::array<uint8_t, 32> seed_b_label = {};
  std::copy_n("B_tensor", 8, seed_b_label.begin());

  TNN_LOG_INFO("%s [FNG] generating dense eal(%dx%d)...\n", tag, h, R);
  const std::vector<int8_t> eal = pearl_dense_noise_reference(h, R, a_noise_seed, seed_a_label);
  TNN_LOG_INFO("%s [FNG] eal size=%zu\n", tag, eal.size());

  TNN_LOG_INFO("%s [FNG] generating sparse ear(%dx%d)...\n", tag, k, R);
  const std::vector<int8_t> ear = pearl_sparse_noise_reference(k, R, a_noise_seed, seed_a_label);
  TNN_LOG_INFO("%s [FNG] ear size=%zu\n", tag, ear.size());

  TNN_LOG_INFO("%s [FNG] generating sparse ebl(%dx%d)...\n", tag, k, R);
  const std::vector<int8_t> ebl = pearl_sparse_noise_reference(k, R, b_noise_seed, seed_b_label);
  TNN_LOG_INFO("%s [FNG] ebl size=%zu\n", tag, ebl.size());

  TNN_LOG_INFO("%s [FNG] generating dense ebr(%dx%d)...\n", tag, w, R);
  const std::vector<int8_t> ebr = pearl_dense_noise_reference(w, R, b_noise_seed, seed_b_label);
  TNN_LOG_INFO("%s [FNG] ebr size=%zu\n", tag, ebr.size());

  TNN_LOG_INFO("%s [FNG] extracting sparse pairs...\n", tag);
  const SparseNoisePairs ear_pairs = extract_sparse_pairs(ear, k, R);
  TNN_LOG_INFO("%s [FNG] ear_pairs extracted: first_idx size=%zu\n", tag, ear_pairs.first_idx.size());
  const SparseNoisePairs ebl_pairs = extract_sparse_pairs(ebl, k, R);
  TNN_LOG_INFO("%s [FNG] ebl_pairs extracted: first_idx size=%zu\n", tag, ebl_pairs.first_idx.size());

  TNN_LOG_INFO("%s [FNG] composing noise_a (%dx%d)...\n", tag, h, k);
  const I32Matrix noise_a = compute_noise_from_factors_dense_times_sparse_T(eal, h, ear_pairs, k, R);
  TNN_LOG_INFO("%s [FNG] noise_a composed: %zu x %zu\n", tag, noise_a.size(), noise_a.empty() ? 0 : noise_a[0].size());

  TNN_LOG_INFO("%s [FNG] composing noise_b (%dx%d)...\n", tag, w, k);
  const I32Matrix noise_b = compute_noise_from_factors_dense_times_sparse_T(ebr, w, ebl_pairs, k, R);
  TNN_LOG_INFO("%s [FNG] noise_b composed: %zu x %zu\n", tag, noise_b.size(), noise_b.empty() ? 0 : noise_b[0].size());

  TNN_LOG_INFO("%s [FNG] computing noised matrices ApEA, BpEB...\n", tag);
  const auto [ApEA, BpEB] = compute_noised_matrices(s_a, s_b, noise_a, noise_b);
  TNN_LOG_INFO("%s [FNG] ApEA size=%zu, BpEB size=%zu\n", tag, ApEA.size(), BpEB.size());

  TNN_LOG_INFO("%s [FNG] computing host ref GEMM (%dx%dx%d)...\n", tag, h, w, k);
  const std::vector<int32_t> C_ref = compute_host_gemm_ref(ApEA, BpEB, h, w, k);
  TNN_LOG_INFO("%s [FNG] C_ref size=%zu, first few: %d %d %d %d\n", tag,
               C_ref.size(),
               C_ref.size() > 0 ? C_ref[0] : 0,
               C_ref.size() > 1 ? C_ref[1] : 0,
               C_ref.size() > 2 ? C_ref[2] : 0,
               C_ref.size() > 3 ? C_ref[3] : 0);

  TNN_LOG_INFO("%s [FNG] compiling fused kernel via RTC...\n", tag);
  auto& compiler = RTCCompiler::instance();
  const std::string fused_source(
    hip_pearl_fused_noise_gemm_source::SRC_TNN_HIP_CRYPTO_PEARL_PEARL_FUSED_NOISE_GEMM_HIP_SOURCE);
  RTCCompiler::CompiledKernel fused_compiled{};
  try
  {
    fused_compiled = compiler.compile_from_source(
      fused_source,
      "pearl_fused_noise_gemm.hip",
      "pearl_fused_noise_gemm",
      compile_opts);
  }
  catch (const std::exception& e)
  {
    TNN_LOG_ERROR("%s [FNG] fused kernel compile failed: %s\n", tag, e.what());
    return false;
  }
  TNN_LOG_INFO("%s [FNG] kernel compiled\n", tag);

  TNN_LOG_INFO("%s [FNG] preparing GPU buffers...\n", tag);

  // Convert sparse pairs to int32_t vectors for GPU upload
  std::vector<int32_t> ear_first_i32(k);
  std::vector<int32_t> ear_second_i32(k);
  std::vector<int32_t> ebl_first_i32(k);
  std::vector<int32_t> ebl_second_i32(k);
  for (int i = 0; i < k; ++i) {
    ear_first_i32[i]  = static_cast<int32_t>(ear_pairs.first_idx[i]);
    ear_second_i32[i] = static_cast<int32_t>(ear_pairs.second_idx[i]);
    ebl_first_i32[i]  = static_cast<int32_t>(ebl_pairs.first_idx[i]);
    ebl_second_i32[i] = static_cast<int32_t>(ebl_pairs.second_idx[i]);
  }

  std::array<uint32_t, 8> s_a_seed_arr = {};
  std::array<uint32_t, 8> s_b_seed_arr = {};

  const std::size_t c_elems = static_cast<std::size_t>(h) * w;

  int32_t* d_ear_first = nullptr;
  int32_t* d_ear_second = nullptr;
  int32_t* d_ebl_first = nullptr;
  int32_t* d_ebl_second = nullptr;
  uint8_t* d_key_a = nullptr;
  uint8_t* d_key_b = nullptr;
  uint8_t* d_seed_a = nullptr;
  uint8_t* d_seed_b = nullptr;
  uint32_t* d_s_a_seed = nullptr;
  uint32_t* d_s_b_seed = nullptr;
  int32_t* d_C = nullptr;

  bool ok = true;
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_ear_first),  k * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_ear_second), k * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_ebl_first),  k * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_ebl_second), k * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_key_a),    a_noise_seed.size()));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_key_b),    b_noise_seed.size()));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_seed_a),   seed_a_label.size()));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_seed_b),   seed_b_label.size()));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_s_a_seed), 8 * sizeof(uint32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_s_b_seed), 8 * sizeof(uint32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_C),        c_elems * sizeof(int32_t)));

  if (!ok)
  {
    TNN_LOG_ERROR("%s [FNG] oroMalloc failed\n", tag);
    (void)oroFree(d_ear_first);  (void)oroFree(d_ear_second);
    (void)oroFree(d_ebl_first);  (void)oroFree(d_ebl_second);
    (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
    (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
    (void)oroFree(d_s_a_seed);   (void)oroFree(d_s_b_seed);
    (void)oroFree(d_C);
    return false;
  }

  ok  = PEARL_GPU_CHECK(oroMemcpy(d_ear_first,  ear_first_i32.data(),  k * sizeof(int32_t), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_ear_second, ear_second_i32.data(), k * sizeof(int32_t), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_ebl_first,  ebl_first_i32.data(),  k * sizeof(int32_t), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_ebl_second, ebl_second_i32.data(), k * sizeof(int32_t), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_key_a,   a_noise_seed.data(),  a_noise_seed.size(),  oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_key_b,   b_noise_seed.data(),  b_noise_seed.size(),  oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_seed_a,  seed_a_label.data(),  seed_a_label.size(),  oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_seed_b,  seed_b_label.data(),  seed_b_label.size(),  oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_s_a_seed, s_a_seed_arr.data(), 8 * sizeof(uint32_t), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_s_b_seed, s_b_seed_arr.data(), 8 * sizeof(uint32_t), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemset(d_C, 0, c_elems * sizeof(int32_t)));

  if (!ok)
  {
    TNN_LOG_ERROR("%s [FNG] oroMemcpy failed\n", tag);
    (void)oroFree(d_ear_first);  (void)oroFree(d_ear_second);
    (void)oroFree(d_ebl_first);  (void)oroFree(d_ebl_second);
    (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
    (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
    (void)oroFree(d_s_a_seed);   (void)oroFree(d_s_b_seed);
    (void)oroFree(d_C);
    return false;
  }

  const int M_gpu = h;
  const int N_gpu = w;
  const int K_gpu = k;
  const int R_gpu = R;
  const int ldd = N_gpu;

  const dim3 gridDim(
    (M_gpu + MACRO_TILE_M - 1) / MACRO_TILE_M,
    (N_gpu + MACRO_TILE_N - 1) / MACRO_TILE_N,
    1);
  const dim3 blockDim(TBLOCK_X, TBLOCK_Y, 1);

  constexpr uint32_t ldsHtA  = MACRO_TILE_M;
  constexpr uint32_t ldsHtB  = MACRO_TILE_N;
  constexpr uint32_t ldsHt   = ldsHtA + ldsHtB;
  constexpr uint32_t ldsW    = MACRO_TILE_K;
  constexpr uint32_t ealSize = MACRO_TILE_M * 32;
  constexpr uint32_t ebrSize = MACRO_TILE_N * 32;
  constexpr uint32_t gemmSize = 2 * ldsHt * ldsW;
  const int sharedMemBytes = static_cast<int>(
    (ealSize + ebrSize + gemmSize) * sizeof(int8_t));

  uint32_t* fused_jackpot_null = nullptr;
  void* kernel_args[] = {
    const_cast<int*>(&M_gpu),
    const_cast<int*>(&N_gpu),
    const_cast<int*>(&K_gpu),
    const_cast<int*>(&R_gpu),
    &d_key_a, &d_key_b, &d_seed_a, &d_seed_b,
    &d_s_a_seed, &d_s_b_seed,
    &d_ear_first, &d_ear_second, &d_ebl_first, &d_ebl_second,
    &d_C, const_cast<int*>(&ldd),
    &fused_jackpot_null
  };

  TNN_LOG_INFO("%s [FNG] launching fused kernel grid=(%u,%u) block=(%u,%u) shmem=%d\n",
               tag, gridDim.x, gridDim.y, blockDim.x, blockDim.y, sharedMemBytes);

  ok = PEARL_GPU_CHECK(oroModuleLaunchKernel(
    fused_compiled.function,
    gridDim.x, gridDim.y, gridDim.z,
    blockDim.x, blockDim.y, blockDim.z,
    sharedMemBytes, nullptr, kernel_args, nullptr));

  ok &= PEARL_GPU_CHECK(oroDeviceSynchronize());

  if (!ok)
  {
    TNN_LOG_ERROR("%s [FNG] Kernel launch or sync failed\n", tag);
    (void)oroFree(d_ear_first);  (void)oroFree(d_ear_second);
    (void)oroFree(d_ebl_first);  (void)oroFree(d_ebl_second);
    (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
    (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
    (void)oroFree(d_s_a_seed);   (void)oroFree(d_s_b_seed);
    (void)oroFree(d_C);
    return false;
  }

  TNN_LOG_INFO("%s [FNG] kernel done, downloading result...\n", tag);

  std::vector<int32_t> hC_fused(c_elems, 0);
  ok = PEARL_GPU_CHECK(oroMemcpy(hC_fused.data(), d_C, c_elems * sizeof(int32_t), oroMemcpyDeviceToHost));
  if (!ok)
  {
    TNN_LOG_ERROR("%s [FNG] oroMemcpy result failed\n", tag);
    (void)oroFree(d_ear_first);  (void)oroFree(d_ear_second);
    (void)oroFree(d_ebl_first);  (void)oroFree(d_ebl_second);
    (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
    (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
    (void)oroFree(d_s_a_seed);   (void)oroFree(d_s_b_seed);
    (void)oroFree(d_C);
    return false;
  }

  (void)oroFree(d_ear_first);  (void)oroFree(d_ear_second);
  (void)oroFree(d_ebl_first);  (void)oroFree(d_ebl_second);
  (void)oroFree(d_C);

  TNN_LOG_INFO("%s [FNG] comparing %d elements against CPU ref...\n", tag, h * w);
  int64_t mismatch_count = 0;
  int first_mismatch_i = -1, first_mismatch_j = -1;
  int32_t first_mismatch_ref = 0, first_mismatch_gpu = 0;

  for (int i = 0; i < h; ++i)
  {
    for (int j = 0; j < w; ++j)
    {
      const size_t idx = static_cast<size_t>(i) * w + j;
      if (C_ref[idx] != hC_fused[idx])
      {
        ++mismatch_count;
        if (first_mismatch_i < 0)
        {
          first_mismatch_i = i;
          first_mismatch_j = j;
          first_mismatch_ref = C_ref[idx];
          first_mismatch_gpu = hC_fused[idx];
        }
      }
    }
  }

  if (mismatch_count == 0)
  {
    TNN_LOG_INFO("%s fused noise GEMM default: PASS (%d elements match exactly)\n", tag, h * w);
  }
  else
  {
    TNN_LOG_ERROR("%s fused noise GEMM default: FAIL\n", tag);
    TNN_LOG_ERROR("%s   total elements: %d\n", tag, h * w);
    TNN_LOG_ERROR("%s   mismatches: %ld\n", tag, static_cast<long>(mismatch_count));
    TNN_LOG_ERROR("%s   first mismatch at (%d, %d): ref=%d gpu=%d\n",
                  tag, first_mismatch_i, first_mismatch_j,
                  first_mismatch_ref, first_mismatch_gpu);
  }

  // ---- Small dimensions test (single block) ----
  TNN_LOG_INFO("%s [FNG] small dimensions test M=128 N=64 K=64 R=32...\n", tag);

  const int h_s = 128, w_s = 64, k_s = 64, R_s = 32;

  I32Matrix s_asmall(h_s, std::vector<int32_t>(k_s));
  I32Matrix s_bsmall(w_s, std::vector<int32_t>(k_s));
  // s_a, s_b elements in [-64, 64] to match Rust SIGNAL_MIN/SIGNAL_MAX
  for (int i = 0; i < h_s; ++i)
    for (int l = 0; l < k_s; ++l)
      s_asmall[i][l] = static_cast<int32_t>(((i * k_s + l) * 0x9e3779b9u) % 129 - 64);
  for (int j = 0; j < w_s; ++j)
    for (int l = 0; l < k_s; ++l)
      s_bsmall[j][l] = static_cast<int32_t>(((j * k_s + l) * 0x9e3779b9u + 0x55) % 129 - 64);

  const std::vector<int8_t> eal_s = pearl_dense_noise_reference(h_s, R_s, a_noise_seed, seed_a_label);
  const std::vector<int8_t> ear_s = pearl_sparse_noise_reference(k_s, R_s, a_noise_seed, seed_a_label);
  const std::vector<int8_t> ebl_s = pearl_sparse_noise_reference(k_s, R_s, b_noise_seed, seed_b_label);
  const std::vector<int8_t> ebr_s = pearl_dense_noise_reference(w_s, R_s, b_noise_seed, seed_b_label);

  const SparseNoisePairs ear_pairs_s = extract_sparse_pairs(ear_s, k_s, R_s);
  const SparseNoisePairs ebl_pairs_s = extract_sparse_pairs(ebl_s, k_s, R_s);

  const I32Matrix noise_asmall = compute_noise_from_factors_dense_times_sparse_T(eal_s, h_s, ear_pairs_s, k_s, R_s);
  const I32Matrix noise_bsmall = compute_noise_from_factors_dense_times_sparse_T(ebr_s, w_s, ebl_pairs_s, k_s, R_s);

  const auto [ApEA_s, BpEB_s] = compute_noised_matrices(s_asmall, s_bsmall, noise_asmall, noise_bsmall);
  const std::vector<int32_t> C_ref_s = compute_host_gemm_ref(ApEA_s, BpEB_s, h_s, w_s, k_s);

  std::vector<int32_t> ear_first_s_i32(k_s);
  std::vector<int32_t> ear_second_s_i32(k_s);
  std::vector<int32_t> ebl_first_s_i32(k_s);
  std::vector<int32_t> ebl_second_s_i32(k_s);
  for (int i = 0; i < k_s; ++i) {
    ear_first_s_i32[i]  = static_cast<int32_t>(ear_pairs_s.first_idx[i]);
    ear_second_s_i32[i] = static_cast<int32_t>(ear_pairs_s.second_idx[i]);
    ebl_first_s_i32[i]  = static_cast<int32_t>(ebl_pairs_s.first_idx[i]);
    ebl_second_s_i32[i]  = static_cast<int32_t>(ebl_pairs_s.second_idx[i]);
  }

  int32_t* ds_ear_first = nullptr;
  int32_t* ds_ear_second = nullptr;
  int32_t* ds_ebl_first = nullptr;
  int32_t* ds_ebl_second = nullptr;
  int32_t* d_C_s = nullptr;

  ok = true;
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&ds_ear_first),  k_s * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&ds_ear_second), k_s * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&ds_ebl_first),  k_s * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&ds_ebl_second), k_s * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_C_s),
    static_cast<std::size_t>(h_s) * w_s * sizeof(int32_t)));

  if (!ok)
  {
    TNN_LOG_ERROR("%s [FNG] small dims oroMalloc failed\n", tag);
    (void)oroFree(ds_ear_first);  (void)oroFree(ds_ear_second);
    (void)oroFree(ds_ebl_first);  (void)oroFree(ds_ebl_second);
    (void)oroFree(d_C_s);
    (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
    (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
    (void)oroFree(d_s_a_seed);   (void)oroFree(d_s_b_seed);
    return false;
  }

  ok  = PEARL_GPU_CHECK(oroMemcpy(ds_ear_first,  ear_first_s_i32.data(),  k_s * sizeof(int32_t), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(ds_ear_second, ear_second_s_i32.data(), k_s * sizeof(int32_t), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(ds_ebl_first,  ebl_first_s_i32.data(),  k_s * sizeof(int32_t), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(ds_ebl_second, ebl_second_s_i32.data(), k_s * sizeof(int32_t), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemset(d_C_s, 0, static_cast<std::size_t>(h_s) * w_s * sizeof(int32_t)));

  if (!ok)
  {
    TNN_LOG_ERROR("%s [FNG] small dims oroMemcpy failed\n", tag);
    (void)oroFree(ds_ear_first);  (void)oroFree(ds_ear_second);
    (void)oroFree(ds_ebl_first);  (void)oroFree(ds_ebl_second);
    (void)oroFree(d_C_s);
    (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
    (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
    (void)oroFree(d_s_a_seed);   (void)oroFree(d_s_b_seed);
    return false;
  }

  {
    const int M_s = h_s, N_s = w_s, K_s = k_s, R_sv = R_s;
    const int ldd_s = N_s;
    const dim3 grid_s(1, 1, 1);
    const int shmem_s = sharedMemBytes;

    uint32_t* jackpot_s_null = nullptr;
    void* args_s[] = {
      const_cast<int*>(&M_s), const_cast<int*>(&N_s),
      const_cast<int*>(&K_s), const_cast<int*>(&R_sv),
      &d_key_a, &d_key_b, &d_seed_a, &d_seed_b,
      &d_s_a_seed, &d_s_b_seed,
      &ds_ear_first, &ds_ear_second, &ds_ebl_first, &ds_ebl_second,
      &d_C_s, const_cast<int*>(&ldd_s),
      &jackpot_s_null
    };

    ok = PEARL_GPU_CHECK(oroModuleLaunchKernel(
      fused_compiled.function,
      grid_s.x, grid_s.y, grid_s.z,
      blockDim.x, blockDim.y, blockDim.z,
      shmem_s, nullptr, args_s, nullptr));
    ok &= PEARL_GPU_CHECK(oroDeviceSynchronize());
  }

  if (!ok)
  {
    TNN_LOG_ERROR("%s [FNG] small dims kernel failed\n", tag);
    (void)oroFree(ds_ear_first);  (void)oroFree(ds_ear_second);
    (void)oroFree(ds_ebl_first);  (void)oroFree(ds_ebl_second);
    (void)oroFree(d_C_s);
    (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
    (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
    (void)oroFree(d_s_a_seed);   (void)oroFree(d_s_b_seed);
    return false;
  }

  std::vector<int32_t> hC_s(static_cast<std::size_t>(h_s) * w_s, 0);
  ok = PEARL_GPU_CHECK(oroMemcpy(hC_s.data(), d_C_s, static_cast<std::size_t>(h_s) * w_s * sizeof(int32_t), oroMemcpyDeviceToHost));

  (void)oroFree(ds_ear_first);  (void)oroFree(ds_ear_second);
  (void)oroFree(ds_ebl_first);  (void)oroFree(ds_ebl_second);
  (void)oroFree(d_C_s);

  if (!ok)
  {
    TNN_LOG_ERROR("%s [FNG] small dims oroMemcpy result failed\n", tag);
    (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
    (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
    (void)oroFree(d_s_a_seed);   (void)oroFree(d_s_b_seed);
    return false;
  }

  int64_t small_mismatch_count = 0;
  for (int i = 0; i < h_s; ++i)
    for (int j = 0; j < w_s; ++j)
      if (C_ref_s[static_cast<size_t>(i) * w_s + j] != hC_s[static_cast<size_t>(i) * w_s + j])
        ++small_mismatch_count;

  if (small_mismatch_count == 0)
  {
    TNN_LOG_INFO("%s fused noise GEMM small: PASS (%d elements match exactly)\n", tag, h_s * w_s);
  }
  else
  {
    TNN_LOG_ERROR("%s fused noise GEMM small: FAIL (%ld mismatches out of %d)\n",
                  tag, static_cast<long>(small_mismatch_count), h_s * w_s);
  }

  (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
  (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
  (void)oroFree(d_s_a_seed);   (void)oroFree(d_s_b_seed);

  return (mismatch_count == 0 && small_mismatch_count == 0);
}

static bool test_pearl_fused_noise_gemm_jackpot(
  const char* tag,
  const std::vector<std::string>& compile_opts)
{
  setvbuf(stdout, NULL, _IONBF, 0);
  TNN_LOG_INFO("%s [JKP] starting fused noise GEMM + jackpot test\n", tag);

  const HarnessFixture fixture = default_fixture();
  TNN_LOG_INFO("%s [JKP] fixture: m=%u n=%u common_dim=%u rank=%u\n", tag,
               fixture.m, fixture.n, fixture.config.common_dim, fixture.config.rank);

  const int h = static_cast<int>(fixture.m);
  const int w = static_cast<int>(fixture.n);
  const int k = static_cast<int>(fixture.config.common_dim);
  const int R = static_cast<int>(fixture.config.rank);

  I32Matrix s_a(h, std::vector<int32_t>(k));
  I32Matrix s_b(w, std::vector<int32_t>(k));
  // s_a, s_b elements in [-64, 64] to match Rust SIGNAL_MIN/SIGNAL_MAX
  for (int i = 0; i < h; ++i)
    for (int l = 0; l < k; ++l)
      s_a[i][l] = static_cast<int32_t>(((i * k + l) * 0x9e3779b9u) % 129 - 64);
  for (int j = 0; j < w; ++j)
    for (int l = 0; l < k; ++l)
      s_b[j][l] = static_cast<int32_t>(((j * k + l) * 0x9e3779b9u + 0x55) % 129 - 64);

  const Hash256 hash_a = hash_from_hex("88c7adb4f1a8d0144fd69ab6a827eba27352a64d506f313c6ceb238a2d50730e");
  const Hash256 hash_b = hash_from_hex("2e3f1797a80ab8028a9524604caad12c036059fcfaceaa601fd95993c5db9034");
  const Hash256 job_key = compute_job_key(fixture.header, fixture.config);
  const auto [b_noise_seed, a_noise_seed] = compute_commitment_hash(job_key, hash_a, hash_b);

  std::array<uint8_t, 32> seed_a_label = {};
  std::copy_n("A_tensor", 8, seed_a_label.begin());
  std::array<uint8_t, 32> seed_b_label = {};
  std::copy_n("B_tensor", 8, seed_b_label.begin());

  const std::vector<int8_t> eal = pearl_dense_noise_reference(h, R, a_noise_seed, seed_a_label);
  const std::vector<int8_t> ear = pearl_sparse_noise_reference(k, R, a_noise_seed, seed_a_label);
  const std::vector<int8_t> ebl = pearl_sparse_noise_reference(k, R, b_noise_seed, seed_b_label);
  const std::vector<int8_t> ebr = pearl_dense_noise_reference(w, R, b_noise_seed, seed_b_label);

  const SparseNoisePairs ear_pairs = extract_sparse_pairs(ear, k, R);
  const SparseNoisePairs ebl_pairs = extract_sparse_pairs(ebl, k, R);

  const I32Matrix noise_a = compute_noise_from_factors_dense_times_sparse_T(eal, h, ear_pairs, k, R);
  const I32Matrix noise_b = compute_noise_from_factors_dense_times_sparse_T(ebr, w, ebl_pairs, k, R);

  const auto [ApEA, BpEB] = compute_noised_matrices(s_a, s_b, noise_a, noise_b);
  const std::vector<int32_t> C_ref = compute_host_gemm_ref(ApEA, BpEB, h, w, k);

  TNN_LOG_INFO("%s [JKP] computing CPU per-tile jackpot XOR...\n", tag);

  const std::vector<uint32_t> cpu_tile_xor = compute_per_tile_xor(
      s_a, s_b, noise_a, noise_b, k, R);
  TNN_LOG_INFO("%s [JKP] CPU tile_xor[0]=%08x tile_xor[16]=%08x\n",
              tag, cpu_tile_xor[0], cpu_tile_xor[16]);
  TNN_LOG_INFO("%s [JKP] CPU per-tile XOR computed: %zu tiles\n", tag, cpu_tile_xor.size());

  TNN_LOG_INFO("%s [JKP] compiling fused kernel via RTC...\n", tag);
  auto& compiler = RTCCompiler::instance();
  const std::string fused_source(
    hip_pearl_fused_noise_gemm_source::SRC_TNN_HIP_CRYPTO_PEARL_PEARL_FUSED_NOISE_GEMM_HIP_SOURCE);
  RTCCompiler::CompiledKernel fused_compiled{};
  try
  {
    fused_compiled = compiler.compile_from_source(
      fused_source,
      "pearl_fused_noise_gemm.hip",
      "pearl_fused_noise_gemm",
      compile_opts);
  }
  catch (const std::exception& e)
  {
    TNN_LOG_ERROR("%s [JKP] fused kernel compile failed: %s\n", tag, e.what());
    return false;
  }
  TNN_LOG_INFO("%s [JKP] kernel compiled\n", tag);

  std::vector<int32_t> ear_first_i32(k);
  std::vector<int32_t> ear_second_i32(k);
  std::vector<int32_t> ebl_first_i32(k);
  std::vector<int32_t> ebl_second_i32(k);
  for (int i = 0; i < k; ++i) {
    ear_first_i32[i]  = static_cast<int32_t>(ear_pairs.first_idx[i]);
    ear_second_i32[i] = static_cast<int32_t>(ear_pairs.second_idx[i]);
    ebl_first_i32[i]  = static_cast<int32_t>(ebl_pairs.first_idx[i]);
    ebl_second_i32[i] = static_cast<int32_t>(ebl_pairs.second_idx[i]);
  }

  std::array<uint32_t, 8> s_a_seed_arr = {};
  std::array<uint32_t, 8> s_b_seed_arr = {};

  const std::size_t c_elems = static_cast<std::size_t>(h) * w;
  const uint32_t num_tiles = static_cast<uint32_t>(k) / static_cast<uint32_t>(R);

  int32_t* d_ear_first = nullptr;
  int32_t* d_ear_second = nullptr;
  int32_t* d_ebl_first = nullptr;
  int32_t* d_ebl_second = nullptr;
  uint8_t* d_key_a = nullptr;
  uint8_t* d_key_b = nullptr;
  uint8_t* d_seed_a = nullptr;
  uint8_t* d_seed_b = nullptr;
  uint32_t* d_s_a_seed = nullptr;
  uint32_t* d_s_b_seed = nullptr;
  int32_t* d_C = nullptr;
  uint32_t* d_jackpot_partial = nullptr;

  bool ok = true;
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_ear_first),   k * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_ear_second),  k * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_ebl_first),   k * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_ebl_second),  k * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_key_a),    a_noise_seed.size()));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_key_b),    b_noise_seed.size()));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_seed_a),   seed_a_label.size()));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_seed_b),   seed_b_label.size()));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_s_a_seed), 8 * sizeof(uint32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_s_b_seed), 8 * sizeof(uint32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_C),        c_elems * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_jackpot_partial), num_tiles * sizeof(uint32_t)));

  if (!ok)
  {
    TNN_LOG_ERROR("%s [JKP] oroMalloc failed\n", tag);
    (void)oroFree(d_ear_first);  (void)oroFree(d_ear_second);
    (void)oroFree(d_ebl_first);  (void)oroFree(d_ebl_second);
    (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
    (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
    (void)oroFree(d_s_a_seed);   (void)oroFree(d_s_b_seed);
    (void)oroFree(d_C);          (void)oroFree(d_jackpot_partial);
    return false;
  }

  ok  = PEARL_GPU_CHECK(oroMemcpy(d_ear_first,  ear_first_i32.data(),  k * sizeof(int32_t), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_ear_second, ear_second_i32.data(), k * sizeof(int32_t), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_ebl_first,  ebl_first_i32.data(),  k * sizeof(int32_t), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_ebl_second, ebl_second_i32.data(), k * sizeof(int32_t), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_key_a,   a_noise_seed.data(),  a_noise_seed.size(),  oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_key_b,   b_noise_seed.data(),  b_noise_seed.size(),  oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_seed_a,  seed_a_label.data(),  seed_a_label.size(),  oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_seed_b,  seed_b_label.data(),  seed_b_label.size(),  oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_s_a_seed, s_a_seed_arr.data(), 8 * sizeof(uint32_t), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(d_s_b_seed, s_b_seed_arr.data(), 8 * sizeof(uint32_t), oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemset(d_C, 0, c_elems * sizeof(int32_t)));
  ok &= PEARL_GPU_CHECK(oroMemset(d_jackpot_partial, 0, num_tiles * sizeof(uint32_t)));

  if (!ok)
  {
    TNN_LOG_ERROR("%s [JKP] oroMemcpy failed\n", tag);
    (void)oroFree(d_ear_first);  (void)oroFree(d_ear_second);
    (void)oroFree(d_ebl_first);  (void)oroFree(d_ebl_second);
    (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
    (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
    (void)oroFree(d_s_a_seed);   (void)oroFree(d_s_b_seed);
    (void)oroFree(d_C);          (void)oroFree(d_jackpot_partial);
    return false;
  }

  const int M_gpu = h;
  const int N_gpu = w;
  const int K_gpu = k;
  const int R_gpu = R;
  const int ldd = N_gpu;

  const dim3 gridDim(
    (M_gpu + MACRO_TILE_M - 1) / MACRO_TILE_M,
    (N_gpu + MACRO_TILE_N - 1) / MACRO_TILE_N,
    1);
  const dim3 blockDim(TBLOCK_X, TBLOCK_Y, 1);

  constexpr uint32_t ldsHtA  = MACRO_TILE_M;
  constexpr uint32_t ldsHtB  = MACRO_TILE_N;
  constexpr uint32_t ldsHt   = ldsHtA + ldsHtB;
  constexpr uint32_t ldsW    = MACRO_TILE_K;
  constexpr uint32_t ealSize = MACRO_TILE_M * 32;
  constexpr uint32_t ebrSize = MACRO_TILE_N * 32;
  constexpr uint32_t gemmSize = 2 * ldsHt * ldsW;
  const int sharedMemBytes = static_cast<int>(
    (ealSize + ebrSize + gemmSize) * sizeof(int8_t));

  void* kernel_args[] = {
    const_cast<int*>(&M_gpu),
    const_cast<int*>(&N_gpu),
    const_cast<int*>(&K_gpu),
    const_cast<int*>(&R_gpu),
    &d_key_a, &d_key_b, &d_seed_a, &d_seed_b,
    &d_s_a_seed, &d_s_b_seed,
    &d_ear_first, &d_ear_second, &d_ebl_first, &d_ebl_second,
    &d_C, const_cast<int*>(&ldd),
    &d_jackpot_partial
  };

  TNN_LOG_INFO("%s [JKP] launching fused+jackpot kernel grid=(%u,%u) block=(%u,%u) shmem=%d\n",
               tag, gridDim.x, gridDim.y, blockDim.x, blockDim.y, sharedMemBytes);

  ok = PEARL_GPU_CHECK(oroModuleLaunchKernel(
    fused_compiled.function,
    gridDim.x, gridDim.y, gridDim.z,
    blockDim.x, blockDim.y, blockDim.z,
    sharedMemBytes, nullptr, kernel_args, nullptr));

  ok &= PEARL_GPU_CHECK(oroDeviceSynchronize());

  if (!ok)
  {
    TNN_LOG_ERROR("%s [JKP] Kernel launch or sync failed\n", tag);
    (void)oroFree(d_ear_first);  (void)oroFree(d_ear_second);
    (void)oroFree(d_ebl_first);  (void)oroFree(d_ebl_second);
    (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
    (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
    (void)oroFree(d_s_a_seed);   (void)oroFree(d_s_b_seed);
    (void)oroFree(d_C);          (void)oroFree(d_jackpot_partial);
    return false;
  }

  TNN_LOG_INFO("%s [JKP] kernel done, downloading results...\n", tag);

  std::vector<int32_t> hC_fused(c_elems, 0);
  std::vector<uint32_t> h_jackpot_partial(num_tiles, 0);
  ok = PEARL_GPU_CHECK(oroMemcpy(hC_fused.data(), d_C, c_elems * sizeof(int32_t), oroMemcpyDeviceToHost));
  ok &= PEARL_GPU_CHECK(oroMemcpy(h_jackpot_partial.data(), d_jackpot_partial, num_tiles * sizeof(uint32_t), oroMemcpyDeviceToHost));

  for (uint32_t t = 0; t < num_tiles; ++t)
      TNN_LOG_INFO("%s [JKP] partial[%u]=%08x\n", tag, t, h_jackpot_partial[t]);

  (void)oroFree(d_ear_first);  (void)oroFree(d_ear_second);
  (void)oroFree(d_ebl_first);  (void)oroFree(d_ebl_second);
  (void)oroFree(d_key_a);      (void)oroFree(d_key_b);
  (void)oroFree(d_seed_a);     (void)oroFree(d_seed_b);
  (void)oroFree(d_s_a_seed);   (void)oroFree(d_s_b_seed);
  (void)oroFree(d_C);          (void)oroFree(d_jackpot_partial);

  if (!ok)
  {
    TNN_LOG_ERROR("%s [JKP] download failed\n", tag);
    return false;
  }

  TNN_LOG_INFO("%s [JKP] comparing C matrix against CPU ref (%d elements)...\n", tag, h * w);
  int64_t mismatch_count = 0;
  int first_mismatch_i = -1, first_mismatch_j = -1;
  for (int i = 0; i < h; ++i)
  {
    for (int j = 0; j < w; ++j)
    {
      const size_t idx = static_cast<size_t>(i) * w + j;
      if (C_ref[idx] != hC_fused[idx])
      {
        ++mismatch_count;
        if (first_mismatch_i < 0)
        {
          first_mismatch_i = i;
          first_mismatch_j = j;
        }
      }
    }
  }

  bool all_pass = true;
  if (mismatch_count == 0)
  {
    TNN_LOG_INFO("%s [JKP] C matrix: PASS (%d elements match)\n", tag, h * w);
  }
  else
  {
    TNN_LOG_ERROR("%s [JKP] C matrix: FAIL (%ld mismatches, first at %d,%d)\n",
                  tag, static_cast<long>(mismatch_count), first_mismatch_i, first_mismatch_j);
    all_pass = false;
  }

  TNN_LOG_INFO("%s [JKP] computing host-side jackpot msg from partial XORs...\n", tag);

  std::array<uint32_t, 16> gpu_jackpot_msg = {};
  for (uint32_t t = 0; t < num_tiles; ++t)
  {
    const uint32_t slot = t % 16u;
    gpu_jackpot_msg[slot] = ((gpu_jackpot_msg[slot] << 13u) | (gpu_jackpot_msg[slot] >> 19u)) ^ h_jackpot_partial[t];
  }

  TNN_LOG_INFO("%s [JKP] computing CPU reference jackpot words...\n", tag);
  const std::array<uint32_t, 16> cpu_jackpot_msg =
    compute_jackpot_words(s_a, s_b, noise_a, noise_b, static_cast<size_t>(k), static_cast<size_t>(R));

  TNN_LOG_INFO("%s [JKP] comparing jackpot msg (16 slots)...\n", tag);
  int64_t jkp_mismatch = 0;
  for (int s = 0; s < 16; ++s)
  {
    if (gpu_jackpot_msg[s] != cpu_jackpot_msg[s])
    {
      ++jkp_mismatch;
      if (jkp_mismatch == 1)
        TNN_LOG_ERROR("%s [JKP] jackpot mismatch at slot %d: GPU=%08x CPU=%08x\n",
                      tag, s, gpu_jackpot_msg[s], cpu_jackpot_msg[s]);
    }
  }

  if (jkp_mismatch == 0)
  {
    TNN_LOG_INFO("%s [JKP] jackpot msg: PASS (all 16 slots match)\n", tag);
  }
  else
  {
    TNN_LOG_ERROR("%s [JKP] jackpot msg: FAIL (%ld slots mismatch)\n",
                  tag, static_cast<long>(jkp_mismatch));
    all_pass = false;
  }

  return all_pass;
}

int test_pearl_hip()
{
#ifdef TNN_HIP
  constexpr const char* tag = "[PEARL-HIP-TEST]";
  TNN_LOG_INFO_COLOR(BRIGHT_CYAN, "%s rocWMMA Cooperative GEMM HIP test\n", tag);

  int device_count = 0;
  if (!PEARL_GPU_CHECK(oroGetDeviceCount(&device_count)))
  {
    TNN_LOG_ERROR("%s orogetDeviceCount failed\n", tag);
    return 1;
  }
  if (device_count == 0)
  {
    TNN_LOG_ERROR("%s No GPU devices found\n", tag);
    return 1;
  }

  oroDeviceProp_t props{};
  if (!PEARL_GPU_CHECK(oroGetDeviceProperties(&props, tnn_get_device(0))))
  {
    TNN_LOG_ERROR("%s oroGetDeviceProperties failed\n", tag);
    return 1;
  }

  oroCtx ctx{};
  if (!PEARL_GPU_CHECK(oroCtxCreate(&ctx, 0, tnn_get_device(0))))
  {
    TNN_LOG_ERROR("%s oroCtxCreate failed\n", tag);
    return 1;
  }

  pearl_register_rtc_headers();
  auto& compiler = RTCCompiler::instance();
  const bool is_amd = tnn_is_amd_device(0);
  const auto compile_opts = pearl_rtc_compile_opts(props, is_amd);

  const std::string kernel_source(
    hip_pearl_gemm_simple_source::SRC_TNN_HIP_CRYPTO_PEARL_PEARL_GEMM_SIMPLE_HIP_SOURCE);

  RTCCompiler::CompiledKernel compiled{};
  try
  {
    compiled = compiler.compile_from_source(
      kernel_source,
      "pearl_gemm_simple.hip",
      "pearl_gemm_simple",
      compile_opts);
  }
  catch (const std::exception& e)
  {
    TNN_LOG_ERROR("%s rocWMMA kernel compile failed: %s\n", tag, e.what());
    (void)oroCtxDestroy(ctx);
    return 1;
  }

  int M = 1024, N = 1024, K = 4096;

  const std::size_t a_elems = static_cast<std::size_t>(M) * K;
  const std::size_t b_elems = static_cast<std::size_t>(K) * N;
  const std::size_t c_elems = static_cast<std::size_t>(M) * N;

  std::vector<signed char> hA(a_elems);
  std::vector<signed char> hB(b_elems);
  for (std::size_t i = 0; i < a_elems; ++i) hA[i] = static_cast<signed char>((i * 0x9e3779b9u) & 0xff);
  for (int j = 0; j < N; ++j)
    for (int l = 0; l < K; ++l)
      hB[static_cast<std::size_t>(j) * K + l] =
        static_cast<signed char>(((static_cast<std::size_t>(j) * K + l) * 0x9e3779b9u + 0x55) & 0xff);

  std::vector<int32_t> hC(c_elems, 0);

  signed char* dA = nullptr;
  signed char* dB = nullptr;
  int32_t* dC = nullptr;

  bool ok = true;
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&dA), a_elems));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&dB), b_elems));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&dC), c_elems * sizeof(int32_t)));

  if (!ok)
  {
    TNN_LOG_ERROR("%s oroMalloc failed\n", tag);
    if (dA) (void)oroFree(dA);
    if (dB) (void)oroFree(dB);
    if (dC) (void)oroFree(dC);
    (void)oroCtxDestroy(ctx);
    return 1;
  }

  ok  = PEARL_GPU_CHECK(oroMemcpy(dA, hA.data(), a_elems, oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemcpy(dB, hB.data(), b_elems, oroMemcpyHostToDevice));
  ok &= PEARL_GPU_CHECK(oroMemset(dC, 0, c_elems * sizeof(int32_t)));

  if (!ok)
  {
    TNN_LOG_ERROR("%s oroMemcpy failed\n", tag);
    (void)oroFree(dA);
    (void)oroFree(dB);
    (void)oroFree(dC);
    (void)oroCtxDestroy(ctx);
    return 1;
  }

  const dim3 gridDim(
    (M + MACRO_TILE_M - 1) / MACRO_TILE_M,
    (N + MACRO_TILE_N - 1) / MACRO_TILE_N,
    1);
  const dim3 blockDim(TBLOCK_X, TBLOCK_Y, 1);

  constexpr uint32_t ldsWidth      = MACRO_TILE_K;
  constexpr uint32_t ldsHeightA    = MACRO_TILE_M;
  constexpr uint32_t ldsHeightB    = MACRO_TILE_N;
  constexpr uint32_t ldsHeight     = ldsHeightA + ldsHeightB;
  constexpr uint32_t sizeLds       = ldsHeight * ldsWidth;
  const int sharedMemBytes = static_cast<int>(2 * sizeLds * sizeof(signed char));

  int lda = K;
  int ldb = K;
  int ldd = N;

  void* kernel_args[] = {&M, &N, &K, &dA, &dB, &dC, &lda, &ldb, &ldd};

  ok = PEARL_GPU_CHECK(oroModuleLaunchKernel(
    compiled.function,
    gridDim.x, gridDim.y, gridDim.z,
    blockDim.x, blockDim.y, blockDim.z,
    sharedMemBytes, nullptr, kernel_args, nullptr));

  ok &= PEARL_GPU_CHECK(oroDeviceSynchronize());

  if (!ok)
  {
    TNN_LOG_ERROR("%s Kernel launch or sync failed\n", tag);
    (void)oroFree(dA);
    (void)oroFree(dB);
    (void)oroFree(dC);
    (void)oroCtxDestroy(ctx);
    return 1;
  }

  ok = PEARL_GPU_CHECK(oroMemcpy(hC.data(), dC, c_elems * sizeof(int32_t), oroMemcpyDeviceToHost));
  if (!ok)
  {
    TNN_LOG_ERROR("%s oroMemcpy result failed\n", tag);
    (void)oroFree(dA);
    (void)oroFree(dB);
    (void)oroFree(dC);
    (void)oroCtxDestroy(ctx);
    return 1;
  }

  bool nonzero = false;
  for (std::size_t i = 0; i < c_elems && !nonzero; ++i)
  {
    if (hC[i] != 0) nonzero = true;
  }

  TNN_LOG_INFO("%s rocWMMA kernel: %s (output %s)\n",
               tag, ok ? "PASSED" : "FAILED",
               nonzero ? "non-zero" : "ALL ZERO");

  (void)oroFree(dA);
  (void)oroFree(dB);
  (void)oroFree(dC);

  bool ng_ok = test_pearl_noised_gemm(tag, compiled);

  bool fused_ok = test_pearl_fused_noise_gemm(tag, compile_opts);

  bool spn_ok = test_pearl_sparse_noise(tag, compile_opts);

  bool jkp_ok = test_pearl_fused_noise_gemm_jackpot(tag, compile_opts);

  (void)oroCtxDestroy(ctx);

  return (ok && nonzero && ng_ok && fused_ok && spn_ok && jkp_ok) ? 0 : 1;
#else
  TNN_LOG_ERROR("[PEARL-HIP-TEST] ERROR: TNN_HIP is not enabled\n");
  return 1;
#endif
}

int bench_pearl_hip()
{
#ifdef TNN_HIP
  constexpr const char* tag = "[PEARL-HIP-BENCH]";
  TNN_LOG_INFO_COLOR(BRIGHT_CYAN, "%s Pearl E2E Mining Pipeline Benchmark\n", tag);

  int device_count = 0;
  if (!PEARL_GPU_CHECK(oroGetDeviceCount(&device_count)))
  {
    TNN_LOG_ERROR("%s oroGetDeviceCount failed\n", tag);
    return 1;
  }
  if (device_count == 0)
  {
    TNN_LOG_ERROR("%s No GPU devices found\n", tag);
    return 1;
  }

  oroDeviceProp_t props{};
  if (!PEARL_GPU_CHECK(oroGetDeviceProperties(&props, tnn_get_device(0))))
  {
    TNN_LOG_ERROR("%s oroGetDeviceProperties failed\n", tag);
    return 1;
  }

  oroCtx ctx{};
  if (!PEARL_GPU_CHECK(oroCtxCreate(&ctx, 0, tnn_get_device(0))))
  {
    TNN_LOG_ERROR("%s oroCtxCreate failed\n", tag);
    return 1;
  }

  pearl_register_rtc_headers();
  auto& compiler = RTCCompiler::instance();
  const bool is_amd = tnn_is_amd_device(0);
  const auto compile_opts = pearl_rtc_compile_opts(props, is_amd);

  const std::string kernel_source(
    hip_pearl_gemm_simple_source::SRC_TNN_HIP_CRYPTO_PEARL_PEARL_GEMM_SIMPLE_HIP_SOURCE);

  RTCCompiler::CompiledKernel compiled{};
  try
  {
    compiled = compiler.compile_from_source(
      kernel_source,
      "pearl_gemm_simple.hip",
      "pearl_gemm_simple",
      compile_opts);
  }
  catch (const std::exception& e)
  {
    TNN_LOG_ERROR("%s rocWMMA kernel compile failed: %s\n", tag, e.what());
    (void)oroCtxDestroy(ctx);
    return 1;
  }

  HarnessFixture fixture = default_fixture();
  const int h = env_or_default("TNN_PEARL_M", static_cast<int>(fixture.m));
  const int w = env_or_default("TNN_PEARL_N", static_cast<int>(fixture.n));
  const int k = env_or_default("TNN_PEARL_K", static_cast<int>(fixture.config.common_dim));
  const int R = env_or_default("TNN_PEARL_R", static_cast<int>(fixture.config.rank));
  const int warmup = env_or_default("TNN_PEARL_WARMUP", 3);
  const int iters  = env_or_default("TNN_PEARL_ITERS", 10);

  TNN_LOG_INFO("%s dims: M=%d N=%d K=%d R=%d warmup=%d iters=%d\n",
               tag, h, w, k, R, warmup, iters);

  I32Matrix s_a(h, std::vector<int32_t>(k));
  I32Matrix s_b(w, std::vector<int32_t>(k));
  // s_a, s_b elements in [-64, 64] to match Rust SIGNAL_MIN/SIGNAL_MAX
  for (int i = 0; i < h; ++i)
    for (int l = 0; l < k; ++l)
      s_a[i][l] = static_cast<int32_t>(((i * k + l) * 0x9e3779b9u) % 129 - 64);
  for (int j = 0; j < w; ++j)
    for (int l = 0; l < k; ++l)
      s_b[j][l] = static_cast<int32_t>(((j * k + l) * 0x9e3779b9u + 0x55) % 129 - 64);

  const Hash256 job_key = compute_job_key(fixture.header, fixture.config);

  std::array<uint8_t, 32> seed_a_label = {};
  std::copy_n("A_tensor", 8, seed_a_label.begin());
  std::array<uint8_t, 32> seed_b_label = {};
  std::copy_n("B_tensor", 8, seed_b_label.begin());

  const std::size_t a_elems = static_cast<std::size_t>(k) * h;
  const std::size_t b_elems = static_cast<std::size_t>(k) * w;
  const std::size_t c_elems = static_cast<std::size_t>(h) * w;

  signed char* dA = nullptr;
  signed char* dB = nullptr;
  int32_t* dC = nullptr;

  bool ok = true;
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&dA), a_elems));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&dB), b_elems));
  ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&dC), c_elems * sizeof(int32_t)));

  if (!ok)
  {
    TNN_LOG_ERROR("%s oroMalloc failed\n", tag);
    if (dA) (void)oroFree(dA);
    if (dB) (void)oroFree(dB);
    if (dC) (void)oroFree(dC);
    (void)oroCtxDestroy(ctx);
    return 1;
  }

  std::vector<signed char> hA(a_elems);
  std::vector<signed char> hB(b_elems);
  std::vector<int32_t> hC(c_elems);

  const dim3 gridDim(
    (h + MACRO_TILE_M - 1) / MACRO_TILE_M,
    (w + MACRO_TILE_N - 1) / MACRO_TILE_N,
    1);
  const dim3 blockDim(TBLOCK_X, TBLOCK_Y, 1);

  constexpr uint32_t ldsWidth    = MACRO_TILE_K;
  constexpr uint32_t ldsHeightA  = MACRO_TILE_M;
  constexpr uint32_t ldsHeightB  = MACRO_TILE_N;
  constexpr uint32_t ldsHeight   = ldsHeightA + ldsHeightB;
  constexpr uint32_t sizeLds     = ldsHeight * ldsWidth;
  const int sharedMemBytes = static_cast<int>(2 * sizeLds * sizeof(signed char));

  hipEvent_t gpu_start{}, gpu_stop{};
  ok = PEARL_GPU_CHECK(hipEventCreate(&gpu_start));
  ok &= PEARL_GPU_CHECK(hipEventCreate(&gpu_stop));

  if (!ok)
  {
    TNN_LOG_ERROR("%s hipEventCreate failed\n", tag);
    (void)oroFree(dA);
    (void)oroFree(dB);
    (void)oroFree(dC);
    (void)oroCtxDestroy(ctx);
    return 1;
  }

  TNN_LOG_INFO("%s warming up (%d iterations)...\n", tag, warmup);
  for (int iter = 0; iter < warmup; ++iter)
  {
    const uint32_t counter = static_cast<uint32_t>(iter);

    uint8_t a_preimage[12] = {};
    std::memcpy(a_preimage, "pearl_a", 7);
    a_preimage[8] = static_cast<uint8_t>(counter & 0xff);
    a_preimage[9] = static_cast<uint8_t>((counter >> 8) & 0xff);
    a_preimage[10] = static_cast<uint8_t>((counter >> 16) & 0xff);
    a_preimage[11] = static_cast<uint8_t>((counter >> 24) & 0xff);
    const Hash256 hash_a = blake3_digest(a_preimage, sizeof(a_preimage));

    uint8_t b_preimage[12] = {};
    std::memcpy(b_preimage, "pearl_b", 7);
    b_preimage[8] = static_cast<uint8_t>(counter & 0xff);
    b_preimage[9] = static_cast<uint8_t>((counter >> 8) & 0xff);
    b_preimage[10] = static_cast<uint8_t>((counter >> 16) & 0xff);
    b_preimage[11] = static_cast<uint8_t>((counter >> 24) & 0xff);
    const Hash256 hash_b = blake3_digest(b_preimage, sizeof(b_preimage));

    const auto [b_noise_seed, a_noise_seed] = compute_commitment_hash(job_key, hash_a, hash_b);

    const std::vector<int8_t> eal = pearl_dense_noise_reference(h, R, a_noise_seed, seed_a_label);
    const std::vector<int8_t> ear = pearl_sparse_noise_reference(k, R, a_noise_seed, seed_a_label);
    const std::vector<int8_t> ebl = pearl_sparse_noise_reference(k, R, b_noise_seed, seed_b_label);
    const std::vector<int8_t> ebr = pearl_dense_noise_reference(w, R, b_noise_seed, seed_b_label);

    const SparseNoisePairs ear_pairs = extract_sparse_pairs(ear, k, R);
    const SparseNoisePairs ebl_pairs = extract_sparse_pairs(ebl, k, R);

    const I32Matrix noise_a = compute_noise_from_factors_dense_times_sparse_T(eal, h, ear_pairs, k, R);
    const I32Matrix noise_b = compute_noise_from_factors_dense_times_sparse_T(ebr, w, ebl_pairs, k, R);

    const auto [ApEA, BpEB] = compute_noised_matrices(s_a, s_b, noise_a, noise_b);

    for (std::size_t i = 0; i < a_elems; ++i) hA[i] = ApEA[i];
    for (std::size_t i = 0; i < b_elems; ++i) hB[i] = BpEB[i];

    ok = PEARL_GPU_CHECK(oroMemcpy(dA, hA.data(), a_elems, oroMemcpyHostToDevice));
    ok &= PEARL_GPU_CHECK(oroMemcpy(dB, hB.data(), b_elems, oroMemcpyHostToDevice));
    ok &= PEARL_GPU_CHECK(oroMemset(dC, 0, c_elems * sizeof(int32_t)));

    int M_gpu = h, N_gpu = w, K_gpu = k;
    int lda = K_gpu, ldb = K_gpu, ldd = N_gpu;
    void* kernel_args[] = {&M_gpu, &N_gpu, &K_gpu, &dA, &dB, &dC, &lda, &ldb, &ldd};

    ok &= PEARL_GPU_CHECK(oroModuleLaunchKernel(
      compiled.function,
      gridDim.x, gridDim.y, gridDim.z,
      blockDim.x, blockDim.y, blockDim.z,
      sharedMemBytes, nullptr, kernel_args, nullptr));

    if (!ok) break;
  }
  ok &= PEARL_GPU_CHECK(oroDeviceSynchronize());

  if (!ok)
  {
    TNN_LOG_ERROR("%s Warmup failed\n", tag);
    (void)hipEventDestroy(gpu_start);
    (void)hipEventDestroy(gpu_stop);
    (void)oroFree(dA);
    (void)oroFree(dB);
    (void)oroFree(dC);
    (void)oroCtxDestroy(ctx);
    return 1;
  }

  TNN_LOG_INFO("%s benchmarking %d iterations...\n", tag, iters);

  std::array<uint32_t, 16> last_jackpot = {};
  Hash256 last_hash_jackpot = {};
  double total_gpu_ms = 0.0;

  using clock = std::chrono::high_resolution_clock;
  const auto wall_start = clock::now();

  for (int iter = 0; iter < iters; ++iter)
  {
    const uint32_t counter = static_cast<uint32_t>(iter + warmup);

    uint8_t a_preimage[12] = {};
    std::memcpy(a_preimage, "pearl_a", 7);
    a_preimage[8] = static_cast<uint8_t>(counter & 0xff);
    a_preimage[9] = static_cast<uint8_t>((counter >> 8) & 0xff);
    a_preimage[10] = static_cast<uint8_t>((counter >> 16) & 0xff);
    a_preimage[11] = static_cast<uint8_t>((counter >> 24) & 0xff);
    const Hash256 hash_a = blake3_digest(a_preimage, sizeof(a_preimage));

    uint8_t b_preimage[12] = {};
    std::memcpy(b_preimage, "pearl_b", 7);
    b_preimage[8] = static_cast<uint8_t>(counter & 0xff);
    b_preimage[9] = static_cast<uint8_t>((counter >> 8) & 0xff);
    b_preimage[10] = static_cast<uint8_t>((counter >> 16) & 0xff);
    b_preimage[11] = static_cast<uint8_t>((counter >> 24) & 0xff);
    const Hash256 hash_b = blake3_digest(b_preimage, sizeof(b_preimage));

    const auto [b_noise_seed, a_noise_seed] = compute_commitment_hash(job_key, hash_a, hash_b);

    const std::vector<int8_t> eal = pearl_dense_noise_reference(h, R, a_noise_seed, seed_a_label);
    const std::vector<int8_t> ear = pearl_sparse_noise_reference(k, R, a_noise_seed, seed_a_label);
    const std::vector<int8_t> ebl = pearl_sparse_noise_reference(k, R, b_noise_seed, seed_b_label);
    const std::vector<int8_t> ebr = pearl_dense_noise_reference(w, R, b_noise_seed, seed_b_label);

    const SparseNoisePairs ear_pairs = extract_sparse_pairs(ear, k, R);
    const SparseNoisePairs ebl_pairs = extract_sparse_pairs(ebl, k, R);

    const I32Matrix noise_a = compute_noise_from_factors_dense_times_sparse_T(eal, h, ear_pairs, k, R);
    const I32Matrix noise_b = compute_noise_from_factors_dense_times_sparse_T(ebr, w, ebl_pairs, k, R);

    const auto [ApEA, BpEB] = compute_noised_matrices(s_a, s_b, noise_a, noise_b);

    for (std::size_t i = 0; i < a_elems; ++i) hA[i] = ApEA[i];
    for (std::size_t i = 0; i < b_elems; ++i) hB[i] = BpEB[i];

    ok = PEARL_GPU_CHECK(oroMemcpy(dA, hA.data(), a_elems, oroMemcpyHostToDevice));
    ok &= PEARL_GPU_CHECK(oroMemcpy(dB, hB.data(), b_elems, oroMemcpyHostToDevice));
    ok &= PEARL_GPU_CHECK(oroMemset(dC, 0, c_elems * sizeof(int32_t)));

    ok &= PEARL_GPU_CHECK(hipEventRecord(gpu_start, nullptr));

    int M_gpu = h, N_gpu = w, K_gpu = k;
    int lda = K_gpu, ldb = K_gpu, ldd = N_gpu;
    void* kernel_args[] = {&M_gpu, &N_gpu, &K_gpu, &dA, &dB, &dC, &lda, &ldb, &ldd};

    ok &= PEARL_GPU_CHECK(oroModuleLaunchKernel(
      compiled.function,
      gridDim.x, gridDim.y, gridDim.z,
      blockDim.x, blockDim.y, blockDim.z,
      sharedMemBytes, nullptr, kernel_args, nullptr));

    ok &= PEARL_GPU_CHECK(hipEventRecord(gpu_stop, nullptr));
    ok &= PEARL_GPU_CHECK(hipEventSynchronize(gpu_stop));

    if (!ok) break;

    float gpu_ms = 0.0f;
    if (PEARL_GPU_CHECK(hipEventElapsedTime(&gpu_ms, gpu_start, gpu_stop)))
      total_gpu_ms += static_cast<double>(gpu_ms);

    ok &= PEARL_GPU_CHECK(oroMemcpy(hC.data(), dC, c_elems * sizeof(int32_t), oroMemcpyDeviceToHost));

    const std::array<uint32_t, 16> jackpot = compute_jackpot_words(s_a, s_b, noise_a, noise_b, k, R);
    const Hash256 hash_jackpot = compute_jackpot_hash(jackpot, a_noise_seed);

    last_jackpot = jackpot;
    last_hash_jackpot = hash_jackpot;
  }

  // ======================================================================
  // Fused noise-GEMM benchmark (pearl_fused_noise_gemm)
  // ======================================================================
  TNN_LOG_INFO("%s compiling fused kernel via RTC...\n", tag);

  const std::string fused_kernel_source(
    hip_pearl_fused_noise_gemm_source::SRC_TNN_HIP_CRYPTO_PEARL_PEARL_FUSED_NOISE_GEMM_HIP_SOURCE);

  RTCCompiler::CompiledKernel fused_compiled{};
  try
  {
    fused_compiled = compiler.compile_from_source(
      fused_kernel_source,
      "pearl_fused_noise_gemm.hip",
      "pearl_fused_noise_gemm",
      compile_opts);
  }
  catch (const std::exception& e)
  {
    TNN_LOG_ERROR("%s fused kernel compile failed: %s\n", tag, e.what());
    // Non-fatal — still report old-path results below
  }

  if (fused_compiled.function)
  {
    // ---- GPU buffers for fused kernel ----
    // keys/seeds: 32 bytes each; s_a_seed/s_b_seed: 32 bytes each (8 u32)
    // sparse pairs: k * sizeof(int32_t) each (4 arrays)
    int32_t* d_fused_ear_first  = nullptr;
    int32_t* d_fused_ear_second = nullptr;
    int32_t* d_fused_ebl_first  = nullptr;
    int32_t* d_fused_ebl_second = nullptr;
    uint8_t* d_fused_key_a      = nullptr;
    uint8_t* d_fused_key_b      = nullptr;
    uint8_t* d_fused_seed_a     = nullptr;
    uint8_t* d_fused_seed_b     = nullptr;
    uint32_t* d_fused_s_a_seed  = nullptr;
    uint32_t* d_fused_s_b_seed  = nullptr;

    bool fused_ok = true;
    bool jackpot_ok = true;
    fused_ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_fused_ear_first),  k * sizeof(int32_t)));
    fused_ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_fused_ear_second), k * sizeof(int32_t)));
    fused_ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_fused_ebl_first),  k * sizeof(int32_t)));
    fused_ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_fused_ebl_second), k * sizeof(int32_t)));
    fused_ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_fused_key_a),      32));
    fused_ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_fused_key_b),      32));
    fused_ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_fused_seed_a),     32));
    fused_ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_fused_seed_b),     32));
    fused_ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_fused_s_a_seed),   8 * sizeof(uint32_t)));
    fused_ok &= PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_fused_s_b_seed),   8 * sizeof(uint32_t)));

    if (!fused_ok)
    {
      TNN_LOG_ERROR("%s fused kernel oroMalloc failed\n", tag);
    }

    // Constant seeds (copy once)
    std::array<uint8_t, 32> seed_a_label_const = {};
    std::copy_n("A_tensor", 8, seed_a_label_const.begin());
    std::array<uint8_t, 32> seed_b_label_const = {};
    std::copy_n("B_tensor", 8, seed_b_label_const.begin());

    fused_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_seed_a, seed_a_label_const.data(), 32, oroMemcpyHostToDevice));
    fused_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_seed_b, seed_b_label_const.data(), 32, oroMemcpyHostToDevice));

    // s_a_seed / s_b_seed (constant per fixture)
    uint32_t h_sa_seed[8], h_sb_seed[8];
    for (int i = 0; i < 8; ++i)
    {
      h_sa_seed[i] = static_cast<uint32_t>(s_a[i < h ? i : 0][0]);
      h_sb_seed[i] = static_cast<uint32_t>(s_b[i < w ? i : 0][0]);
    }
    fused_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_s_a_seed, h_sa_seed, 8 * sizeof(uint32_t), oroMemcpyHostToDevice));
    fused_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_s_b_seed, h_sb_seed, 8 * sizeof(uint32_t), oroMemcpyHostToDevice));

    // Shared memory size for fused kernel
    constexpr uint32_t fused_ealSize = MACRO_TILE_M * 32;
    constexpr uint32_t fused_ebrSize = MACRO_TILE_N * 32;
    constexpr uint32_t fused_ldsHt   = MACRO_TILE_M + MACRO_TILE_N;
    constexpr uint32_t fused_ldsW    = MACRO_TILE_K;
    constexpr uint32_t fused_gemmSize = 2 * fused_ldsHt * fused_ldsW;
    const int fused_sharedMemBytes = static_cast<int>(
      (fused_ealSize + fused_ebrSize + fused_gemmSize) * sizeof(int8_t));

    const dim3 fused_gridDim(
      (h + MACRO_TILE_M - 1) / MACRO_TILE_M,
      (w + MACRO_TILE_N - 1) / MACRO_TILE_N,
      1);
    const dim3 fused_blockDim(TBLOCK_X, TBLOCK_Y, 1);

    hipEvent_t fused_start{}, fused_stop{};
    fused_ok &= PEARL_GPU_CHECK(hipEventCreate(&fused_start));
    fused_ok &= PEARL_GPU_CHECK(hipEventCreate(&fused_stop));

    // ---- Fused warmup ----
    if (fused_ok)
    {
      // Pre-compute ear/ebl sparse pairs for warmup + benchmark
      // (these change per iteration — computed on CPU each iteration)
      TNN_LOG_INFO("%s warming up fused kernel (%d iterations)...\n", tag, warmup);
      for (int iter = 0; iter < warmup; ++iter)
      {
        const uint32_t counter = static_cast<uint32_t>(iter);
        uint8_t a_preimage[12] = {};
        std::memcpy(a_preimage, "pearl_a", 7);
        a_preimage[8] = static_cast<uint8_t>(counter & 0xff);
        a_preimage[9] = static_cast<uint8_t>((counter >> 8) & 0xff);
        a_preimage[10] = static_cast<uint8_t>((counter >> 16) & 0xff);
        a_preimage[11] = static_cast<uint8_t>((counter >> 24) & 0xff);
        const Hash256 hash_a = blake3_digest(a_preimage, sizeof(a_preimage));
        uint8_t b_preimage[12] = {};
        std::memcpy(b_preimage, "pearl_b", 7);
        b_preimage[8] = static_cast<uint8_t>(counter & 0xff);
        b_preimage[9] = static_cast<uint8_t>((counter >> 8) & 0xff);
        b_preimage[10] = static_cast<uint8_t>((counter >> 16) & 0xff);
        b_preimage[11] = static_cast<uint8_t>((counter >> 24) & 0xff);
        const Hash256 hash_b = blake3_digest(b_preimage, sizeof(b_preimage));
        const auto [b_noise_seed, a_noise_seed] = compute_commitment_hash(job_key, hash_a, hash_b);

        // Sparse noise (ear/ebl) — still needed on CPU for index pairs
        const std::vector<int8_t> ear = pearl_sparse_noise_reference(k, R, a_noise_seed, seed_a_label_const);
        const std::vector<int8_t> ebl = pearl_sparse_noise_reference(k, R, b_noise_seed, seed_b_label_const);
        const SparseNoisePairs ear_pairs = extract_sparse_pairs(ear, k, R);
        const SparseNoisePairs ebl_pairs = extract_sparse_pairs(ebl, k, R);

        std::vector<int32_t> ear_first(k), ear_second(k);
        std::vector<int32_t> ebl_first(k), ebl_second(k);
        for (int i = 0; i < k; ++i) {
          ear_first[i]  = static_cast<int32_t>(ear_pairs.first_idx[i]);
          ear_second[i] = static_cast<int32_t>(ear_pairs.second_idx[i]);
          ebl_first[i]  = static_cast<int32_t>(ebl_pairs.first_idx[i]);
          ebl_second[i] = static_cast<int32_t>(ebl_pairs.second_idx[i]);
        }

        fused_ok = PEARL_GPU_CHECK(oroMemcpy(d_fused_ear_first,  ear_first.data(),  k * sizeof(int32_t), oroMemcpyHostToDevice));
        fused_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_ear_second, ear_second.data(), k * sizeof(int32_t), oroMemcpyHostToDevice));
        fused_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_ebl_first,  ebl_first.data(),  k * sizeof(int32_t), oroMemcpyHostToDevice));
        fused_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_ebl_second, ebl_second.data(), k * sizeof(int32_t), oroMemcpyHostToDevice));
        fused_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_key_a, a_noise_seed.data(), 32, oroMemcpyHostToDevice));
        fused_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_key_b, b_noise_seed.data(), 32, oroMemcpyHostToDevice));
        fused_ok &= PEARL_GPU_CHECK(oroMemset(dC, 0, c_elems * sizeof(int32_t)));

        if (!fused_ok) break;

        const int M_gpu = h, N_gpu = w, K_gpu = k, R_gpu = R, ldd_gpu = N_gpu;
        uint32_t* fused_jackpot_null = nullptr;
        void* fused_args[] = {
          const_cast<int*>(&M_gpu), const_cast<int*>(&N_gpu),
          const_cast<int*>(&K_gpu), const_cast<int*>(&R_gpu),
          &d_fused_key_a, &d_fused_key_b,
          &d_fused_seed_a, &d_fused_seed_b,
          &d_fused_s_a_seed, &d_fused_s_b_seed,
          &d_fused_ear_first, &d_fused_ear_second,
          &d_fused_ebl_first, &d_fused_ebl_second,
          &dC, const_cast<int*>(&ldd_gpu),
          &fused_jackpot_null
        };

        fused_ok &= PEARL_GPU_CHECK(oroModuleLaunchKernel(
          fused_compiled.function,
          fused_gridDim.x, fused_gridDim.y, fused_gridDim.z,
          fused_blockDim.x, fused_blockDim.y, fused_blockDim.z,
          fused_sharedMemBytes, nullptr, fused_args, nullptr));
      }
      fused_ok &= PEARL_GPU_CHECK(oroDeviceSynchronize());
    }

    // ---- Fused benchmark ----
    double total_fused_gpu_ms = 0.0;
    double total_jackpot_gpu_ms = 0.0;
    double total_jackpot_wall_ms = 0.0;
    const auto fused_wall_start = clock::now();

    for (int iter = 0; fused_ok && iter < iters; ++iter)
    {
      const uint32_t counter = static_cast<uint32_t>(iter + warmup);
      uint8_t a_preimage[12] = {};
      std::memcpy(a_preimage, "pearl_a", 7);
      a_preimage[8] = static_cast<uint8_t>(counter & 0xff);
      a_preimage[9] = static_cast<uint8_t>((counter >> 8) & 0xff);
      a_preimage[10] = static_cast<uint8_t>((counter >> 16) & 0xff);
      a_preimage[11] = static_cast<uint8_t>((counter >> 24) & 0xff);
      const Hash256 hash_a = blake3_digest(a_preimage, sizeof(a_preimage));
      uint8_t b_preimage[12] = {};
      std::memcpy(b_preimage, "pearl_b", 7);
      b_preimage[8] = static_cast<uint8_t>(counter & 0xff);
      b_preimage[9] = static_cast<uint8_t>((counter >> 8) & 0xff);
      b_preimage[10] = static_cast<uint8_t>((counter >> 16) & 0xff);
      b_preimage[11] = static_cast<uint8_t>((counter >> 24) & 0xff);
      const Hash256 hash_b = blake3_digest(b_preimage, sizeof(b_preimage));
      const auto [b_noise_seed, a_noise_seed] = compute_commitment_hash(job_key, hash_a, hash_b);

      const std::vector<int8_t> ear = pearl_sparse_noise_reference(k, R, a_noise_seed, seed_a_label_const);
      const std::vector<int8_t> ebl = pearl_sparse_noise_reference(k, R, b_noise_seed, seed_b_label_const);
      const SparseNoisePairs ear_pairs = extract_sparse_pairs(ear, k, R);
      const SparseNoisePairs ebl_pairs = extract_sparse_pairs(ebl, k, R);

      std::vector<int32_t> ear_first(k), ear_second(k);
      std::vector<int32_t> ebl_first(k), ebl_second(k);
      for (int i = 0; i < k; ++i) {
        ear_first[i]  = static_cast<int32_t>(ear_pairs.first_idx[i]);
        ear_second[i] = static_cast<int32_t>(ear_pairs.second_idx[i]);
        ebl_first[i]  = static_cast<int32_t>(ebl_pairs.first_idx[i]);
        ebl_second[i] = static_cast<int32_t>(ebl_pairs.second_idx[i]);
      }

      fused_ok = PEARL_GPU_CHECK(oroMemcpy(d_fused_ear_first,  ear_first.data(),  k * sizeof(int32_t), oroMemcpyHostToDevice));
      fused_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_ear_second, ear_second.data(), k * sizeof(int32_t), oroMemcpyHostToDevice));
      fused_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_ebl_first,  ebl_first.data(),  k * sizeof(int32_t), oroMemcpyHostToDevice));
      fused_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_ebl_second, ebl_second.data(), k * sizeof(int32_t), oroMemcpyHostToDevice));
      fused_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_key_a, a_noise_seed.data(), 32, oroMemcpyHostToDevice));
      fused_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_key_b, b_noise_seed.data(), 32, oroMemcpyHostToDevice));
      fused_ok &= PEARL_GPU_CHECK(oroMemset(dC, 0, c_elems * sizeof(int32_t)));

      if (!fused_ok) break;

      const int M_gpu = h, N_gpu = w, K_gpu = k, R_gpu = R, ldd_gpu = N_gpu;
      uint32_t* fused_jackpot_null = nullptr;
      void* fused_args[] = {
        const_cast<int*>(&M_gpu), const_cast<int*>(&N_gpu),
        const_cast<int*>(&K_gpu), const_cast<int*>(&R_gpu),
        &d_fused_key_a, &d_fused_key_b,
        &d_fused_seed_a, &d_fused_seed_b,
        &d_fused_s_a_seed, &d_fused_s_b_seed,
        &d_fused_ear_first, &d_fused_ear_second,
        &d_fused_ebl_first, &d_fused_ebl_second,
        &dC, const_cast<int*>(&ldd_gpu),
        &fused_jackpot_null
      };

      fused_ok &= PEARL_GPU_CHECK(hipEventRecord(fused_start, nullptr));
      fused_ok &= PEARL_GPU_CHECK(oroModuleLaunchKernel(
        fused_compiled.function,
        fused_gridDim.x, fused_gridDim.y, fused_gridDim.z,
        fused_blockDim.x, fused_blockDim.y, fused_blockDim.z,
        fused_sharedMemBytes, nullptr, fused_args, nullptr));
      fused_ok &= PEARL_GPU_CHECK(hipEventRecord(fused_stop, nullptr));
      fused_ok &= PEARL_GPU_CHECK(hipEventSynchronize(fused_stop));

      if (!fused_ok) break;

      float fused_ms = 0.0f;
      if (PEARL_GPU_CHECK(hipEventElapsedTime(&fused_ms, fused_start, fused_stop)))
        total_fused_gpu_ms += static_cast<double>(fused_ms);

      fused_ok &= PEARL_GPU_CHECK(oroMemcpy(hC.data(), dC, c_elems * sizeof(int32_t), oroMemcpyDeviceToHost));
    }

    const auto fused_wall_end = clock::now();
    const double fused_wall_ms = static_cast<double>(
      std::chrono::duration_cast<std::chrono::microseconds>(fused_wall_end - fused_wall_start).count()) / 1000.0;

    (void)hipEventDestroy(fused_start);
    (void)hipEventDestroy(fused_stop);

    // ---- Fused+jackpot benchmark ----
    {
      const uint32_t bench_num_tiles = static_cast<uint32_t>(k) / static_cast<uint32_t>(R);
      uint32_t* d_jackpot_partial = nullptr;
      jackpot_ok = PEARL_GPU_CHECK(oroMalloc(reinterpret_cast<oroDeviceptr*>(&d_jackpot_partial), bench_num_tiles * sizeof(uint32_t)));
      jackpot_ok &= PEARL_GPU_CHECK(oroMemset(d_jackpot_partial, 0, bench_num_tiles * sizeof(uint32_t)));

      hipEvent_t jackpot_start{}, jackpot_stop{};
      jackpot_ok &= PEARL_GPU_CHECK(hipEventCreate(&jackpot_start));
      jackpot_ok &= PEARL_GPU_CHECK(hipEventCreate(&jackpot_stop));

      if (jackpot_ok)
      {
        TNN_LOG_INFO("%s warming up fused+jackpot kernel (%d iterations)...\n", tag, warmup);
        for (int iter = 0; iter < warmup; ++iter)
        {
          const uint32_t counter = static_cast<uint32_t>(iter);
          uint8_t a_preimage[12] = {};
          std::memcpy(a_preimage, "pearl_a", 7);
          a_preimage[8] = static_cast<uint8_t>(counter & 0xff);
          a_preimage[9] = static_cast<uint8_t>((counter >> 8) & 0xff);
          a_preimage[10] = static_cast<uint8_t>((counter >> 16) & 0xff);
          a_preimage[11] = static_cast<uint8_t>((counter >> 24) & 0xff);
          const Hash256 hash_a = blake3_digest(a_preimage, sizeof(a_preimage));
          uint8_t b_preimage[12] = {};
          std::memcpy(b_preimage, "pearl_b", 7);
          b_preimage[8] = static_cast<uint8_t>(counter & 0xff);
          b_preimage[9] = static_cast<uint8_t>((counter >> 8) & 0xff);
          b_preimage[10] = static_cast<uint8_t>((counter >> 16) & 0xff);
          b_preimage[11] = static_cast<uint8_t>((counter >> 24) & 0xff);
          const Hash256 hash_b = blake3_digest(b_preimage, sizeof(b_preimage));
          const auto [b_noise_seed, a_noise_seed] = compute_commitment_hash(job_key, hash_a, hash_b);

          const std::vector<int8_t> ear = pearl_sparse_noise_reference(k, R, a_noise_seed, seed_a_label_const);
          const std::vector<int8_t> ebl = pearl_sparse_noise_reference(k, R, b_noise_seed, seed_b_label_const);
          const SparseNoisePairs ear_pairs = extract_sparse_pairs(ear, k, R);
          const SparseNoisePairs ebl_pairs = extract_sparse_pairs(ebl, k, R);

          std::vector<int32_t> ear_first(k), ear_second(k);
          std::vector<int32_t> ebl_first(k), ebl_second(k);
          for (int i = 0; i < k; ++i) {
            ear_first[i]  = static_cast<int32_t>(ear_pairs.first_idx[i]);
            ear_second[i] = static_cast<int32_t>(ear_pairs.second_idx[i]);
            ebl_first[i]  = static_cast<int32_t>(ebl_pairs.first_idx[i]);
            ebl_second[i] = static_cast<int32_t>(ebl_pairs.second_idx[i]);
          }

          jackpot_ok = PEARL_GPU_CHECK(oroMemcpy(d_fused_ear_first,  ear_first.data(),  k * sizeof(int32_t), oroMemcpyHostToDevice));
          jackpot_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_ear_second, ear_second.data(), k * sizeof(int32_t), oroMemcpyHostToDevice));
          jackpot_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_ebl_first,  ebl_first.data(),  k * sizeof(int32_t), oroMemcpyHostToDevice));
          jackpot_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_ebl_second, ebl_second.data(), k * sizeof(int32_t), oroMemcpyHostToDevice));
          jackpot_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_key_a, a_noise_seed.data(), 32, oroMemcpyHostToDevice));
          jackpot_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_key_b, b_noise_seed.data(), 32, oroMemcpyHostToDevice));
          jackpot_ok &= PEARL_GPU_CHECK(oroMemset(dC, 0, c_elems * sizeof(int32_t)));
          jackpot_ok &= PEARL_GPU_CHECK(oroMemset(d_jackpot_partial, 0, bench_num_tiles * sizeof(uint32_t)));

          if (!jackpot_ok) break;

          const int M_gpu = h, N_gpu = w, K_gpu = k, R_gpu = R, ldd_gpu = N_gpu;
          void* jackpot_args[] = {
            const_cast<int*>(&M_gpu), const_cast<int*>(&N_gpu),
            const_cast<int*>(&K_gpu), const_cast<int*>(&R_gpu),
            &d_fused_key_a, &d_fused_key_b,
            &d_fused_seed_a, &d_fused_seed_b,
            &d_fused_s_a_seed, &d_fused_s_b_seed,
            &d_fused_ear_first, &d_fused_ear_second,
            &d_fused_ebl_first, &d_fused_ebl_second,
            &dC, const_cast<int*>(&ldd_gpu),
            &d_jackpot_partial
          };

          jackpot_ok &= PEARL_GPU_CHECK(oroModuleLaunchKernel(
            fused_compiled.function,
            fused_gridDim.x, fused_gridDim.y, fused_gridDim.z,
            fused_blockDim.x, fused_blockDim.y, fused_blockDim.z,
            fused_sharedMemBytes, nullptr, jackpot_args, nullptr));
        }
        jackpot_ok &= PEARL_GPU_CHECK(oroDeviceSynchronize());
      }

      if (jackpot_ok)
      {
        TNN_LOG_INFO("%s benchmarking fused+jackpot %d iterations...\n", tag, iters);
        const auto jackpot_wall_start = clock::now();

        for (int iter = 0; iter < iters; ++iter)
        {
          const uint32_t counter = static_cast<uint32_t>(iter + warmup);
          uint8_t a_preimage[12] = {};
          std::memcpy(a_preimage, "pearl_a", 7);
          a_preimage[8] = static_cast<uint8_t>(counter & 0xff);
          a_preimage[9] = static_cast<uint8_t>((counter >> 8) & 0xff);
          a_preimage[10] = static_cast<uint8_t>((counter >> 16) & 0xff);
          a_preimage[11] = static_cast<uint8_t>((counter >> 24) & 0xff);
          const Hash256 hash_a = blake3_digest(a_preimage, sizeof(a_preimage));
          uint8_t b_preimage[12] = {};
          std::memcpy(b_preimage, "pearl_b", 7);
          b_preimage[8] = static_cast<uint8_t>(counter & 0xff);
          b_preimage[9] = static_cast<uint8_t>((counter >> 8) & 0xff);
          b_preimage[10] = static_cast<uint8_t>((counter >> 16) & 0xff);
          b_preimage[11] = static_cast<uint8_t>((counter >> 24) & 0xff);
          const Hash256 hash_b = blake3_digest(b_preimage, sizeof(b_preimage));
          const auto [b_noise_seed, a_noise_seed] = compute_commitment_hash(job_key, hash_a, hash_b);

          const std::vector<int8_t> ear = pearl_sparse_noise_reference(k, R, a_noise_seed, seed_a_label_const);
          const std::vector<int8_t> ebl = pearl_sparse_noise_reference(k, R, b_noise_seed, seed_b_label_const);
          const SparseNoisePairs ear_pairs = extract_sparse_pairs(ear, k, R);
          const SparseNoisePairs ebl_pairs = extract_sparse_pairs(ebl, k, R);

          std::vector<int32_t> ear_first(k), ear_second(k);
          std::vector<int32_t> ebl_first(k), ebl_second(k);
          for (int i = 0; i < k; ++i) {
            ear_first[i]  = static_cast<int32_t>(ear_pairs.first_idx[i]);
            ear_second[i] = static_cast<int32_t>(ear_pairs.second_idx[i]);
            ebl_first[i]  = static_cast<int32_t>(ebl_pairs.first_idx[i]);
            ebl_second[i] = static_cast<int32_t>(ebl_pairs.second_idx[i]);
          }

          jackpot_ok = PEARL_GPU_CHECK(oroMemcpy(d_fused_ear_first,  ear_first.data(),  k * sizeof(int32_t), oroMemcpyHostToDevice));
          jackpot_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_ear_second, ear_second.data(), k * sizeof(int32_t), oroMemcpyHostToDevice));
          jackpot_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_ebl_first,  ebl_first.data(),  k * sizeof(int32_t), oroMemcpyHostToDevice));
          jackpot_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_ebl_second, ebl_second.data(), k * sizeof(int32_t), oroMemcpyHostToDevice));
          jackpot_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_key_a, a_noise_seed.data(), 32, oroMemcpyHostToDevice));
          jackpot_ok &= PEARL_GPU_CHECK(oroMemcpy(d_fused_key_b, b_noise_seed.data(), 32, oroMemcpyHostToDevice));
          jackpot_ok &= PEARL_GPU_CHECK(oroMemset(dC, 0, c_elems * sizeof(int32_t)));
          jackpot_ok &= PEARL_GPU_CHECK(oroMemset(d_jackpot_partial, 0, bench_num_tiles * sizeof(uint32_t)));

          if (!jackpot_ok) break;

          const int M_gpu = h, N_gpu = w, K_gpu = k, R_gpu = R, ldd_gpu = N_gpu;
          void* jackpot_args[] = {
            const_cast<int*>(&M_gpu), const_cast<int*>(&N_gpu),
            const_cast<int*>(&K_gpu), const_cast<int*>(&R_gpu),
            &d_fused_key_a, &d_fused_key_b,
            &d_fused_seed_a, &d_fused_seed_b,
            &d_fused_s_a_seed, &d_fused_s_b_seed,
            &d_fused_ear_first, &d_fused_ear_second,
            &d_fused_ebl_first, &d_fused_ebl_second,
            &dC, const_cast<int*>(&ldd_gpu),
            &d_jackpot_partial
          };

          jackpot_ok &= PEARL_GPU_CHECK(hipEventRecord(jackpot_start, nullptr));
          jackpot_ok &= PEARL_GPU_CHECK(oroModuleLaunchKernel(
            fused_compiled.function,
            fused_gridDim.x, fused_gridDim.y, fused_gridDim.z,
            fused_blockDim.x, fused_blockDim.y, fused_blockDim.z,
            fused_sharedMemBytes, nullptr, jackpot_args, nullptr));
          jackpot_ok &= PEARL_GPU_CHECK(hipEventRecord(jackpot_stop, nullptr));
          jackpot_ok &= PEARL_GPU_CHECK(hipEventSynchronize(jackpot_stop));

          if (!jackpot_ok) break;

          float jackpot_ms = 0.0f;
          if (PEARL_GPU_CHECK(hipEventElapsedTime(&jackpot_ms, jackpot_start, jackpot_stop)))
            total_jackpot_gpu_ms += static_cast<double>(jackpot_ms);

          jackpot_ok &= PEARL_GPU_CHECK(oroMemcpy(hC.data(), dC, c_elems * sizeof(int32_t), oroMemcpyDeviceToHost));
        }

        const auto jackpot_wall_end = clock::now();
        total_jackpot_wall_ms = static_cast<double>(
          std::chrono::duration_cast<std::chrono::microseconds>(jackpot_wall_end - jackpot_wall_start).count()) / 1000.0;
      }
      else
      {
        TNN_LOG_ERROR("%s fused+jackpot setup failed\n", tag);
      }

      (void)hipEventDestroy(jackpot_start);
      (void)hipEventDestroy(jackpot_stop);
      (void)oroFree(d_jackpot_partial);
    }

    // ---- Fused cleanup ----
    (void)oroFree(d_fused_ear_first);
    (void)oroFree(d_fused_ear_second);
    (void)oroFree(d_fused_ebl_first);
    (void)oroFree(d_fused_ebl_second);
    (void)oroFree(d_fused_key_a);
    (void)oroFree(d_fused_key_b);
    (void)oroFree(d_fused_seed_a);
    (void)oroFree(d_fused_seed_b);
    (void)oroFree(d_fused_s_a_seed);
    (void)oroFree(d_fused_s_b_seed);

    // ---- Results ----
    const auto wall_end = clock::now();
    const double total_wall_ms = static_cast<double>(
      std::chrono::duration_cast<std::chrono::microseconds>(wall_end - wall_start).count()) / 1000.0;

    (void)hipEventDestroy(gpu_start);
    (void)hipEventDestroy(gpu_stop);

    if (!ok)
    {
      TNN_LOG_ERROR("%s Benchmark iteration failed\n", tag);
      (void)oroFree(dA);
      (void)oroFree(dB);
      (void)oroFree(dC);
      (void)oroCtxDestroy(ctx);
      return 1;
    }

    const double ms_per_eval       = total_wall_ms / static_cast<double>(iters);
    const double evals_per_sec     = (ms_per_eval > 0.0) ? (1000.0 / ms_per_eval) : 0.0;
    const double avg_gpu_ms        = total_gpu_ms / static_cast<double>(iters);
    const double avg_cpu_ms        = ms_per_eval - avg_gpu_ms;
    const double fused_per_eval    = fused_ok ? fused_wall_ms / static_cast<double>(iters) : 0.0;
    const double fused_evals_sec   = (fused_per_eval > 0.0) ? (1000.0 / fused_per_eval) : 0.0;
    const double avg_fused_gpu_ms  = fused_ok ? total_fused_gpu_ms / static_cast<double>(iters) : 0.0;
    const double avg_fused_cpu_ms  = fused_per_eval - avg_fused_gpu_ms;
    const double jackpot_per_eval  = jackpot_ok ? total_jackpot_wall_ms / static_cast<double>(iters) : 0.0;
    const double jackpot_evals_sec = (jackpot_per_eval > 0.0) ? (1000.0 / jackpot_per_eval) : 0.0;
    const double avg_jackpot_gpu_ms = jackpot_ok ? total_jackpot_gpu_ms / static_cast<double>(iters) : 0.0;
    const double avg_jackpot_cpu_ms = jackpot_per_eval - avg_jackpot_gpu_ms;
    const double speedup           = (fused_per_eval > 0.0) ? ms_per_eval / fused_per_eval : 0.0;

    TNN_LOG_INFO("%s === Side-by-Side Benchmark Results ===\n", tag);
    TNN_LOG_INFO("%s dims:        M=%d N=%d K=%d R=%d  iters=%d  warmup=%d\n", tag, h, w, k, R, iters, warmup);
    TNN_LOG_INFO("%s %-28s %12s %12s %12s\n", tag, "Metric", "gemm_simple", "fused_noise", "fused+jackpot");
    TNN_LOG_INFO("%s %-28s %12s %12s %12s\n", tag, "-----", "-----------", "-----------", "-------------");
    TNN_LOG_INFO("%s %-28s %12.3f %12.3f %12.3f  ms\n", tag, "per eval (wall)", ms_per_eval, fused_per_eval, jackpot_per_eval);
    TNN_LOG_INFO("%s %-28s %12.3f %12.3f %12.3f  ms\n", tag, "  GPU (kernel only)", avg_gpu_ms, avg_fused_gpu_ms, avg_jackpot_gpu_ms);
    TNN_LOG_INFO("%s %-28s %12.3f %12.3f %12.3f  ms\n", tag, "  CPU (noise+compose+setup)", avg_cpu_ms, avg_fused_cpu_ms, avg_jackpot_cpu_ms);
    TNN_LOG_INFO("%s %-28s %12.2f %12.2f %12.2f  eval/s\n", tag, "hashrate", evals_per_sec, fused_evals_sec, jackpot_evals_sec);
    TNN_LOG_INFO("%s %-28s %12.2fx\n", tag, "speedup (fused vs gemm_simple)", speedup);
    TNN_LOG_INFO("%s last jackpot hash: %s\n", tag, hex_bytes(last_hash_jackpot).c_str());

    (void)oroFree(dA);
    (void)oroFree(dB);
    (void)oroFree(dC);
    (void)oroCtxDestroy(ctx);

    return (ok && fused_ok) ? 0 : 1;
  }
  else
  {
    // Fused kernel not available — fall back to old-path-only results
    const auto wall_end = clock::now();
    const double total_wall_ms = static_cast<double>(
      std::chrono::duration_cast<std::chrono::microseconds>(wall_end - wall_start).count()) / 1000.0;

    (void)hipEventDestroy(gpu_start);
    (void)hipEventDestroy(gpu_stop);

    if (!ok)
    {
      TNN_LOG_ERROR("%s Benchmark iteration failed\n", tag);
      (void)oroFree(dA);
      (void)oroFree(dB);
      (void)oroFree(dC);
      (void)oroCtxDestroy(ctx);
      return 1;
    }

    const double ms_per_eval   = total_wall_ms / static_cast<double>(iters);
    const double evals_per_sec = (ms_per_eval > 0.0) ? (1000.0 / ms_per_eval) : 0.0;
    const double avg_gpu_ms    = total_gpu_ms / static_cast<double>(iters);
    const double avg_cpu_ms    = ms_per_eval - avg_gpu_ms;

    TNN_LOG_INFO("%s === Benchmark Results (gemm_simple only) ===\n", tag);
    TNN_LOG_INFO("%s dims:        M=%d N=%d K=%d R=%d\n", tag, h, w, k, R);
    TNN_LOG_INFO("%s iterations:  %d  warmup: %d\n", tag, iters, warmup);
    TNN_LOG_INFO("%s per eval:    %9.3f ms\n", tag, ms_per_eval);
    TNN_LOG_INFO("%s   GPU:       %9.3f ms (kernel only)\n", tag, avg_gpu_ms);
    TNN_LOG_INFO("%s   CPU:       %9.3f ms (noise+sX+gEM+jackpot)\n", tag, avg_cpu_ms);
    TNN_LOG_INFO("%s hashrate:    %9.2f eval/s\n", tag, evals_per_sec);
    TNN_LOG_INFO("%s last jackpot hash: %s\n", tag, hex_bytes(last_hash_jackpot).c_str());

    (void)oroFree(dA);
    (void)oroFree(dB);
    (void)oroFree(dC);
    (void)oroCtxDestroy(ctx);

    return ok ? 0 : 1;
  }
#else
  TNN_LOG_ERROR("[PEARL-HIP-BENCH] ERROR: TNN_HIP is not enabled\n");
  return 1;
#endif
}

} // namespace tnn::pearl
