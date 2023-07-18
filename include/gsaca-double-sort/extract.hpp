#pragma once

#include "uint_types.hpp"

namespace gsaca_lyndon {

template<typename count_type, typename value_type>
gsaca_always_inline uint128_t
extract(value_type const *const text, count_type const &idx,
        uint8_t const count) {
  uint128_t result = text[idx];
  for (count_type i = 1; i < count; ++i) {
    result <<= 8;
    result |= text[idx + i];
  }
  return result;
}

template<typename count_type, typename value_type>
gsaca_always_inline uint128_t
safe_extract(value_type const *const text, count_type const &idx,
             uint8_t const count) {
  uint128_t result = text[idx];
  count_type i;
  for (i = 1; i < count; ++i) {
    if (text[idx + i] == 0) break; // sentinel at end of text!
    result <<= (sizeof(value_type)*8);
    result |= text[idx + i];
  }
  uint8_t const shift = sizeof(value_type) == 1 ? 3 : 5;
  return result << ((count - i) << shift);
}

}
