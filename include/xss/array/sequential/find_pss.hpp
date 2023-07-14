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

#include "xss/common/util.hpp"

namespace xss {
namespace internal {

  template <typename ctx_type, typename index_type>
  xss_always_inline static void xss_array_find_pss(const ctx_type& ctx,
                                                   const index_type j,
                                                   const index_type i,
                                                   const index_type lce,
                                                   index_type& max_lce_j,
                                                   index_type& max_lce,
                                                   index_type& pss_of_i) {
    index_type upper = j;
    index_type upper_lce = lce;
    index_type lower = upper;
    index_type lower_lce = 0;

    while (ctx.text[upper + upper_lce] > ctx.text[i + upper_lce]) {
      if (xss_unlikely(lower == upper)) {
        for (index_type k = 0; k < upper_lce; ++k)
          lower = ctx.array[lower];
        lower_lce = ctx.get_lce.with_upper_bound(lower, i, upper_lce);
      } else {
        lower_lce =
            ctx.get_lce.with_both_bounds(lower, i, lower_lce, upper_lce);
      }
      if (xss_unlikely(lower_lce == upper_lce)) {
        upper = ctx.array[upper];
        upper_lce = ctx.get_lce.with_lower_bound(upper, i, upper_lce);
      } else
        break;
    }

    // if at this point we have (upper == lower), then we also have
    // text[upper + upper_lce] < text[i + upper_lce]
    if (ctx.text[upper + upper_lce] < ctx.text[i + upper_lce]) {
      // PSS of i is upper
      max_lce_j = pss_of_i = upper;
      max_lce = upper_lce;
    } else {
      // PSS of i lies between upper and lower (could be lower, but not upper)
      // we definitely have upper > lower
      index_type upper_idx = ctx.n - 1;
      index_type lower_idx = upper_idx;
      ctx.array[upper_idx] = upper;
      while (upper > lower) {
        ctx.array[--lower_idx] = ctx.array[upper];
        upper = ctx.array[upper];
      }
      upper = ctx.array[upper_idx];

      while (true) {
        // move lower until same LCE as upper
        lower_lce = ctx.get_lce.with_both_bounds(ctx.array[lower_idx], i,
                                                 lower_lce, upper_lce);
        while (lower_lce < upper_lce) {
          ++lower_idx;
          lower_lce = ctx.get_lce.with_both_bounds(ctx.array[lower_idx], i,
                                                   lower_lce, upper_lce);
        }

        if (lower_idx == upper_idx) {
          pss_of_i = ctx.array[lower_idx - 1];
          break;
        }

        --upper_idx;
        upper_lce =
            ctx.get_lce.with_lower_bound(ctx.array[upper_idx], i, upper_lce);

        if (ctx.text[ctx.array[upper_idx] + upper_lce] <
            ctx.text[i + upper_lce]) {
          pss_of_i = ctx.array[upper_idx];
          break;
        }
      }

      max_lce_j = ctx.array[upper_idx];
      max_lce = upper_lce;
    }
  }

} // namespace internal
} // namespace xss