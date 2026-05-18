#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace tnn::pearl {

using Hash256 = std::array<uint8_t, 32>;

constexpr size_t kIncompleteBlockHeaderBytes = 76;
constexpr size_t kMiningConfigBytes = 52;
constexpr size_t kHashBytes = 32;
constexpr size_t kPatternDims = 3;
constexpr size_t kMiningConfigReservedBytes = 32;
constexpr uint16_t kMmaInt7xInt7ToInt32 = 0;

struct IncompleteBlockHeader {
  uint32_t version = 0;
  Hash256 prev_block{};
  Hash256 merkle_root{};
  uint32_t timestamp = 0;
  uint32_t nbits = 0;

  std::array<uint8_t, kIncompleteBlockHeaderBytes> to_bytes() const;
};

struct PeriodicPattern {
  std::array<std::pair<uint32_t, uint32_t>, kPatternDims> shape{};

  std::array<uint8_t, 2 * kPatternDims> to_bytes() const;
  std::vector<uint32_t> to_list() const;
  uint32_t period() const;
  uint32_t size() const;
};

struct MiningConfiguration {
  uint32_t common_dim = 0;
  uint16_t rank = 0;
  uint16_t mma_type = kMmaInt7xInt7ToInt32;
  PeriodicPattern rows_pattern{};
  PeriodicPattern cols_pattern{};
  std::array<uint8_t, kMiningConfigReservedBytes> reserved{};

  std::array<uint8_t, kMiningConfigBytes> to_bytes() const;
  uint32_t dot_product_length() const;
};

struct HarnessFixture {
  const char* name = nullptr;
  uint32_t m = 0;
  uint32_t n = 0;
  IncompleteBlockHeader header{};
  MiningConfiguration config{};
};

Hash256 blake3_digest(const uint8_t* data, size_t len);
Hash256 blake3_keyed_digest(const uint8_t* data, size_t len, const Hash256& key);
Hash256 compute_job_key(const IncompleteBlockHeader& header, const MiningConfiguration& config);
std::pair<Hash256, Hash256> compute_commitment_hash(const Hash256& job_key, const Hash256& hash_a, const Hash256& hash_b);
Hash256 compute_jackpot_hash(const std::array<uint32_t, 16>& jackpot, const Hash256& a_noise_seed);

HarnessFixture default_fixture();

} // namespace tnn::pearl
