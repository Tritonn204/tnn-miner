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

  template <typename index_type, typename value_type>
  struct lce_type {
    const value_type* text;

    xss_always_inline index_type without_bounds(const index_type l,
                                                const index_type r,
                                                index_type lce = 0) const {
      while (text[l + lce] == text[r + lce])
        ++lce;
      return lce;
    }

    xss_always_inline index_type
    with_both_bounds(const index_type l,
                     const index_type r,
                     index_type lower,
                     const index_type upper) const {
      while (lower < upper && text[l + lower] == text[r + lower])
        ++lower;
      return lower;
    }

    xss_always_inline index_type with_upper_bound(
        const index_type l, const index_type r, const index_type upper) const {
      return with_both_bounds(l, r, 0, upper);
    }

    xss_always_inline index_type with_lower_bound(
        const index_type l, const index_type r, const index_type lower) const {
      return without_bounds(l, r, lower);
    }
  };

} // namespace internal
} // namespace xss