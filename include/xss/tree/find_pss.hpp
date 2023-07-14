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

#include "stack.hpp"
#include "xss/common/util.hpp"

namespace xss {
namespace internal {

  template <typename ctx_type, typename index_type>
  xss_always_inline static void pss_tree_find_pss(ctx_type& ctx,
                                                  const index_type j,
                                                  const index_type i,
                                                  const index_type lce,
                                                  index_type& max_lce_j,
                                                  index_type& max_lce,
                                                  index_type& pss_of_i) {

    if (ctx.text[j + lce] < ctx.text[i + lce]) {
      max_lce = lce;
      max_lce_j = pss_of_i = j;
      return;
    }

    // reverse stack will contains elements that might be the PSS of i
    auto& stack = ctx.stack;
    reverse_telescope_stack reverse_stack;
    uint64_t rev_stack_size = 0;

    index_type new_j = j;
    index_type new_lce = lce;

    while (ctx.text[new_j + new_lce] > ctx.text[i + new_lce]) {
      // new_j = stack.top() is not the pss.
      max_lce_j = new_j;
      max_lce = new_lce;

      // everything on the rev stack is not the pss
      while (reverse_stack.top() < reverse_telescope_stack::max_val) {
        reverse_stack.pop();
        ctx.stream.append_closing_parenthesis();
      }

      // look up to new_lce many elements into the future
      rev_stack_size = 0;
      for (; rev_stack_size < new_lce && stack.top() > 0; ++rev_stack_size) {
        reverse_stack.push(stack.top());
        stack.pop();
      }

      if (rev_stack_size == 0) {
        pss_of_i = 0;
        return;
      }

      new_j = reverse_stack.top();
      new_lce = ctx.get_lce.without_bounds(new_j, i);
    }

    //    std::cout << "RSS: " << rev_stack_size << " new_j: " << new_j <<
    //    std::endl;

    // now the PSS is contained in the reverse stack (or it is 0)
    // find it with binary search!
    while (rev_stack_size > 1) {
      const uint64_t half_size = (rev_stack_size >> 1);
      for (uint64_t k = 0; k < half_size; ++k) {
        stack.push(reverse_stack.top());
        reverse_stack.pop();
      }

      // check if PSS still on reverse stack
      new_j = reverse_stack.top();
      new_lce = ctx.get_lce.without_bounds(new_j, i);
      if (ctx.text[new_j + new_lce] < ctx.text[i + new_lce]) {
        // PSS is still on reverse stack
        rev_stack_size -= half_size;
        continue;
      } else {
        // PSS is not on reverse stack
        max_lce_j = new_j;
        max_lce = new_lce;
        while (reverse_stack.top() < reverse_telescope_stack::max_val) {
          reverse_stack.pop();
          ctx.stream.append_closing_parenthesis();
        }
        for (uint64_t k = 0; k < half_size; ++k) {
          reverse_stack.push(stack.top());
          stack.pop();
        }
        rev_stack_size = half_size;
      }
    }

    // now the PSS is the only element on the reverse stack (or it is 0)
    new_j = reverse_stack.top();
    new_lce = ctx.get_lce.without_bounds(new_j, i);
    if (ctx.text[new_j + new_lce] < ctx.text[i + new_lce]) {
      // pss = new_j
      pss_of_i = new_j;
      if (new_lce >= max_lce) {
        max_lce_j = new_j;
        max_lce = new_lce;
      }
      stack.push(pss_of_i);
    } else {
      // pss = 0
      pss_of_i = 0;
      max_lce_j = new_j;
      max_lce = new_lce;
      ctx.stream.append_closing_parenthesis();
    }
    //    std::cout << "RSS: " << rev_stack_size << " pss: " << pss_of_i <<
    //    std::endl;
  }

} // namespace internal
} // namespace xss