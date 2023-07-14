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

#include "lce.hpp"
#include "util.hpp"
#include "xss/tree/bit_vector.hpp"

namespace xss {
namespace internal {

  template <typename index_type, typename value_type>
  struct array_context_type {

    const value_type* text;
    index_type* array;
    const index_type n;

    index_type* aux = nullptr;

    const lce_type<index_type, value_type> get_lce =
        lce_type<index_type, value_type>{text};
  };

  template <typename stack_type, typename index_type, typename value_type>
  struct tree_context_type {

    const value_type* text;
    bit_vector& bv;
    parentheses_stream& stream;
    stack_type& stack;
    const index_type n;

    const lce_type<index_type, value_type> get_lce =
        lce_type<index_type, value_type>{text};
  };

} // namespace internal
} // namespace xss