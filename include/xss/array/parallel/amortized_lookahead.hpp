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

  template <typename ctx_type, typename index_type>
  xss_always_inline static void
  pss_array_amortized_lookahead(ctx_type& ctx,
                                const index_type j,
                                index_type& i,
                                const index_type max_lce,
                                const index_type distance,
                                const index_type upper) {
    const index_type anchor =
        std::min(get_anchor(&(ctx.text[i]), max_lce), upper - i);
    // copy values up to anchor
    for (index_type k = 1; k < anchor; ++k) {
      ctx.array[i + k] = ctx.array[j + k] + distance;
    }
    i += anchor - 1;
  }

} // namespace internal
} // namespace xss