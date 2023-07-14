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

#include "duval.hpp"
#include "util.hpp"

namespace xss {
namespace internal {

  template <typename index_type, typename value_type>
  xss_always_inline index_type get_anchor(const value_type* lce_str,
                                          const index_type lce_len) {

    const index_type ell = lce_len >> 2;

    // check if gamm_ell is an extended lyndon run
    const auto duval = is_extended_lyndon_run(&(lce_str[ell]), lce_len - ell);

    // try to extend the lyndon run as far as possible to the left
    if (duval.first > 0) {
      const index_type period = duval.first;
      const auto repetition_eq = [&](const index_type l, const index_type r) {
        for (index_type k = 0; k < period; ++k)
          if (lce_str[l + k] != lce_str[r + k])
            return false;
        return true;
      };
      int64_t lhs = ell + duval.second - period;
      while (lhs >= 0 && repetition_eq(lhs, lhs + period)) {
        lhs -= period;
      }
      return std::min(ell, (index_type)(lhs + (period << 1)));
    } else {
      return ell;
    }
  }

} // namespace internal
} // namespace xss