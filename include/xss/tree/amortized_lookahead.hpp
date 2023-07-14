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
  pss_tree_amortized_lookahead(ctx_type& ctx,
                               const index_type j,
                               index_type& i,
                               index_type lce,
                               const index_type distance) {

    bool j_smaller_i = ctx.text[j + lce] < ctx.text[i + lce];
    const index_type anchor = get_anchor(&(ctx.text[i]), lce);
    const uint64_t bps_distance = 2 * distance - ((j_smaller_i) ? (1) : (0));

    if (bps_distance <= 64)
      return;

    auto& stack = ctx.stack;
    auto& bv = ctx.bv;
    auto& stream = ctx.stream;
    uint64_t bps_idx = stream.bits_written() - bps_distance;
    uint64_t count_open = 0;

    while (count_open < anchor - 1) {
      if (bv.get(bps_idx++)) {
        stream.append_opening_parenthesis();
        count_open++;
        stack.push(i + count_open);
      } else {
        stream.append_closing_parenthesis();
        stack.pop();
      }
    }

    i += anchor - 1;
  }

} // namespace internal
} // namespace xss