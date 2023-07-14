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

#include "amortized_lookahead.hpp"
#include "bit_vector.hpp"
#include "find_pss.hpp"
#include "run_extension.hpp"
#include "stack.hpp"
#include "xss/common/util.hpp"

namespace xss {

template <typename index_type = uint64_t, typename value_type>
static void pss_tree(value_type const* const text,
                     uint64_t* const result_data,
                     uint64_t const n,
                     uint64_t threshold = internal::DEFAULT_THRESHOLD) {
  using namespace internal;
  using stack_type = buffered_stack<telescope_stack, index_type>;
  warn_type_width<index_type>(n, "xss::pss_tree");
  fix_threshold(threshold);

  bit_vector result(result_data, (n << 1) + 2);
  parentheses_stream stream(result);
  stack_type stack(n >> 3, telescope_stack());
  tree_context_type<stack_type, index_type, value_type> ctx{
      text, result, stream, stack, (index_type) n};

  // open node 0;
  stream.append_opening_parenthesis();
  stream.append_opening_parenthesis();

  index_type j, lce;
  for (index_type i = 1; i < n - 1; ++i) {
    j = i - 1; // = stack.top();
    lce = ctx.get_lce.without_bounds(j, i);

    if (xss_likely(lce <= threshold)) {
      while (text[j + lce] > text[i + lce]) {
        stack.pop();
        j = stack.top();
        stream.append_closing_parenthesis();
        lce = ctx.get_lce.without_bounds(j, i);
        if (xss_unlikely(lce > threshold))
          break;
      }
    }

    if (xss_likely(lce <= threshold)) {
      stack.push(i);
      stream.append_opening_parenthesis();
      continue;
    }

    index_type max_lce = 0, max_lce_j = 0, pss_of_i = 0;
    pss_tree_find_pss(ctx, j, i, lce, max_lce_j, max_lce, pss_of_i);

    stack.push(i);
    stream.append_opening_parenthesis();

    const index_type distance = i - max_lce_j;
    if (xss_unlikely(max_lce >= 2 * distance))
      pss_tree_run_extension(ctx, max_lce_j, i, max_lce, distance);
    else
      pss_tree_amortized_lookahead(ctx, max_lce_j, i, max_lce, distance);
  }

  while (stack.top() > 0) {
    stack.pop();
    stream.append_closing_parenthesis();
  }
  stream.append_closing_parenthesis();
  stream.append_opening_parenthesis();
  stream.append_closing_parenthesis();
  stream.append_closing_parenthesis();
}

} // namespace xss
