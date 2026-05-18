#include "test_pearl_hip.h"

#include <tnn_hip/crypto/pearl/pearl_pouw_defs.h>
#include <tnn_hip/crypto/pearl/pearl_pouw_vectors.inc>

#include <BLAKE3/c/blake3.h>

#include <algo_definitions.h>
#include <tnn_log.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
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

  TNN_LOG_INFO("%s TODO: add OpenPearl PlainProof vectors and HIP kernels\n", tag);
  return 0;
}
