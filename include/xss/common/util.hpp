//  Copyright (c) 2019 Jonas Ellert
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
//  sell copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

#pragma once

#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#define xss_always_inline __attribute__((always_inline)) inline
#define xss_likely(x) __builtin_expect(!!(x), 1)
#define xss_unlikely(x) __builtin_expect(!!(x), 0)

namespace xss {
namespace internal {

  // can never be below 8
  constexpr static uint64_t MIN_THRESHOLD = 8;
  constexpr static uint64_t DEFAULT_THRESHOLD = 128;

  inline static void fix_threshold(uint64_t& threshold) {
    threshold = std::max(threshold, MIN_THRESHOLD);
  }

  template <typename index_type>
  static void warn_type_width(const uint64_t n, const std::string name) {
    if (n > std::numeric_limits<index_type>::max()) {
      std::cerr << "WARNING: " << name << " --- n=" << n
                << ": Given index_type of width " << sizeof(index_type)
                << " bytes is insufficient!" << std::endl;
    }
  }

} // namespace internal
} // namespace xss