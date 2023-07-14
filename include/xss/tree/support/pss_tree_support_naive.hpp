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

#include "xss/tree/bit_vector.hpp"

namespace xss {

class pss_tree_support_naive {
private:
  const bit_vector bv_;
  std::vector<uint64_t> select_open_;
  std::vector<uint64_t> select_close_;

public:
  pss_tree_support_naive(const uint64_t* data, const uint64_t bits)
      : bv_(const_cast<uint64_t*>(data), bits),
        select_open_((bits >> 1) + 1),
        select_close_((bits >> 1) + 1) {
    uint64_t ones = 0;
    uint64_t zeros = 0;
    for (uint64_t i = 0; i < bv_.size(); ++i) {
      if (bv_.get(i))
        select_open_[++ones] = i;
      else
        select_close_[++zeros] = i;
    }
  }

  template <typename bv_type>
  pss_tree_support_naive(const bv_type& bv)
      : pss_tree_support_naive(bv.data(), bv.size()) {}

  xss_always_inline uint64_t enclose(const uint64_t bps_idx) const {
    uint64_t result = bps_idx;
    uint64_t excess = (bv_.get(result)) ? 1 : 2;
    while (excess > 0) {
      if (bv_.get(--result))
        --excess;
      else
        ++excess;
    }
    return result;
  }

  xss_always_inline uint64_t find_close(const uint64_t bps_idx) const {
    uint64_t result = bps_idx;
    uint64_t excess = (bv_.get(result)) ? 1 : 0;
    while (excess > 0) {
      if (bv_.get(++result))
        ++excess;
      else
        --excess;
    }
    return result;
  }

  xss_always_inline uint64_t parent_distance(const uint64_t bps_idx) const {
    const uint64_t bps_idx_open_parent = enclose(bps_idx);
    return (bps_idx - bps_idx_open_parent + 1) >> 1;
  }

  xss_always_inline uint64_t subtree_size(const uint64_t bps_idx) const {
    const uint64_t bps_idx_close_nss = find_close(bps_idx);
    return (bps_idx_close_nss - bps_idx + 1) >> 1;
  }

  xss_always_inline uint64_t pss(const uint64_t preorder_number) const {
    const uint64_t bps_idx_open_node = select_open_[preorder_number + 2];
    const uint64_t parent_dist = parent_distance(bps_idx_open_node);
    return preorder_number - parent_dist;
  }

  xss_always_inline uint64_t nss(const uint64_t preorder_number) const {
    const uint64_t bps_idx_open_node = select_open_[preorder_number + 2];
    const uint64_t subtree = subtree_size(bps_idx_open_node);
    return preorder_number + subtree;
  }

  xss_always_inline uint64_t lyndon(const uint64_t preorder_number) const {
    return nss(preorder_number) - preorder_number;
  }
};

} // namespace xss