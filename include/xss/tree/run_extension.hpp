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
  xss_always_inline static void
  pss_tree_run_extension(ctx_type& ctx,
                         const index_type j,
                         index_type& i,
                         index_type lce,
                         const index_type period) {
    bool j_smaller_i = ctx.text[j + lce] < ctx.text[i + lce];
    const uint64_t bps_distance = 2 * period - ((j_smaller_i) ? (1) : (0));
    const index_type repetitions = lce / period - 1;

    //    std::cout << "RE " << j << " " << i << " " << lce << " " <<
    //    j_smaller_i
    //              << " " << repetitions << " " << bps_distance << std::flush;

    i += (repetitions * period);
    if (j_smaller_i) {
      for (uint64_t r = 0; r < repetitions; ++r) {
        ctx.stack.push(ctx.stack.top() + period);
      }
    } else {
      ctx.stack.pop();
      ctx.stack.push(i);
    }

    ctx.stream.append_copy(bps_distance, bps_distance * repetitions);
    //    std::cout << " DONE. next i: " << i << std::endl;
  }

} // namespace internal
} // namespace xss