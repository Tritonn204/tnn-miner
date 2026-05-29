#include <tnn_hip/crypto/pearl/pearl_pouw_defs.h>

#include <algorithm>
#include <cstdint>
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

} // anonymous namespace

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

} // namespace tnn::pearl
