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

#include "xss/common/anchor.hpp"
#include "xss/common/util.hpp"

namespace xss {

namespace internal {

  template <bool build_nss,
            bool build_lyndon,
            typename ctx_type,
            typename index_type>
  xss_always_inline static void
  pss_array_amortized_lookahead(ctx_type& ctx,
                                const index_type j,
                                index_type& i,
                                index_type max_lce,
                                const index_type distance) {
    const index_type anchor = get_anchor(&(ctx.text[i]), max_lce);
    // copy NSS values up to anchor
    for (index_type k = 1; k < anchor; ++k) {
      ctx.array[i + k] = ctx.array[j + k] + distance;
      if constexpr (build_nss)
        ctx.aux[i + k] = ctx.aux[j + k] + distance;
      if constexpr (build_lyndon)
        ctx.aux[i + k] = ctx.aux[j + k];
    }
    i += anchor - 1;
  }

  template <typename ctx_type, typename index_type>
  xss_always_inline static void
  nss_array_amortized_lookahead(ctx_type& ctx,
                                const index_type j,
                                index_type& i,
                                index_type max_lce,
                                const index_type distance) {

    const index_type anchor = get_anchor(&(ctx.text[i]), max_lce);
    index_type next_pss = i;
    // copy NSS values up to anchor
    for (index_type k = 1; k < anchor; ++k) {
      if (ctx.array[j + k] < j + anchor) {
        ctx.array[i + k] = ctx.array[j + k] + distance;
      } else {
        ctx.array[i + k] = next_pss;
        next_pss = i + k;
      }
    }
    i += anchor - 1;
  }

  template <typename ctx_type, typename index_type>
  xss_always_inline static void lyndon_array_amortized_lookahead(
      ctx_type& ctx, const index_type j, index_type& i, index_type max_lce) {

    const index_type anchor = get_anchor(&(ctx.text[i]), max_lce);
    index_type next_pss = i;
    // copy NSS values up to anchor
    for (index_type k = 1; k < anchor; ++k) {
      if (ctx.array[j + k] + j + k < j + anchor) {
        ctx.array[i + k] = ctx.array[j + k];
      } else {
        ctx.array[i + k] = next_pss;
        next_pss = i + k;
      }
    }
    i += anchor - 1;
  }

} // namespace internal
} // namespace xss