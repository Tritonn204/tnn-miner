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
#include <sdsl/bp_support_sada.hpp>

namespace xss {

class pss_tree_support_sdsl {
private:
  using bv_type = sdsl::bit_vector;
  using support_type = sdsl::bp_support_sada<>;

  const bv_type& bv_;
  const support_type support_;

public:
  pss_tree_support_sdsl(const bv_type& bv) : bv_(bv), support_(&bv_) {}

  xss_always_inline uint64_t parent_distance(const uint64_t bps_idx) const {
    const uint64_t bps_idx_open_parent = support_.enclose(bps_idx);
    return (bps_idx - bps_idx_open_parent + 1) >> 1;
  }

  xss_always_inline uint64_t subtree_size(const uint64_t bps_idx) const {
    const uint64_t bps_idx_close_nss = support_.find_close(bps_idx);
    return (bps_idx_close_nss - bps_idx + 1) >> 1;
  }

  xss_always_inline uint64_t pss(const uint64_t preorder_number) const {
    const uint64_t bps_idx_open_node = support_.select(preorder_number + 2);
    const uint64_t parent_dist = parent_distance(bps_idx_open_node);
    return preorder_number - parent_dist;
  }

  xss_always_inline uint64_t nss(const uint64_t preorder_number) const {
    const uint64_t bps_idx_open_node = support_.select(preorder_number + 2);
    const uint64_t subtree = subtree_size(bps_idx_open_node);
    return preorder_number + subtree;
  }

  xss_always_inline uint64_t lyndon(const uint64_t preorder_number) const {
    return nss(preorder_number) - preorder_number;
  }
};

} // namespace xss