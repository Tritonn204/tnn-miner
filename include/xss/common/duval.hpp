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

#include "util.hpp"

namespace xss {
namespace internal {

  template <typename value_type>
  xss_always_inline static std::pair<uint64_t, uint64_t>
  is_extended_lyndon_run(const value_type* text, const uint64_t n) {
    std::pair<uint64_t, uint64_t> result = {0, 0};
    uint64_t i = 0;
    while (i < n) {
      uint64_t j = i + 1, k = i;
      while (j < n && text[k] <= text[j]) {
        if (text[k] < text[j])
          k = i;
        else
          k++;
        j++;
      }
      if (xss_unlikely((j - k) > result.first)) {
        result.first = j - k;
        result.second = i;
      }
      while (i <= k) {
        i += j - k;
      }
    }
    const uint64_t period = result.first;
    if (2 * period > n)
      return {0, 0};
    for (i = period; i < n; ++i) {
      if (xss_unlikely(text[i - period] != text[i]))
        return {0, 0};
    }
    return result;
  }

} // namespace internal
} // namespace xss