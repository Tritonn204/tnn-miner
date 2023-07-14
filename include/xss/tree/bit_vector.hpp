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
#include <cstring>
#include <sstream>

namespace xss {

class bit_vector {
private:
  uint64_t n_bits_;
  uint64_t n_words_;
  uint64_t n_bytes_;
  uint64_t* data_;
  uint64_t* delete_data_;

  xss_always_inline static uint64_t mod64(const uint64_t idx) {
    return idx - ((idx >> 6) << 6);
  }

public:
  bit_vector(const uint64_t n)
      : n_bits_(n),
        n_words_((n_bits_ + 127) >> 6),
        n_bytes_(n_words_ << 3),
        data_(static_cast<uint64_t*>(malloc(n_bytes_))),
        delete_data_(data_) {}

  bit_vector(const uint64_t n, bool default_value) : bit_vector(n) {
    if (default_value)
      memset(data_, -1, n_bytes_);
    else
      memset(data_, 0, n_bytes_);
  }

  bit_vector(uint64_t* data, uint64_t n)
      : n_bits_(n),
        n_words_((n_bits_ + 63) >> 6),
        n_bytes_(n_words_ << 3),
        data_(data),
        delete_data_(nullptr){};

  xss_always_inline void set_one(const uint64_t idx) {
    data_[idx >> 6] |= 1ULL << mod64(idx);
  }

  xss_always_inline void set_zero(const uint64_t idx) {
    data_[idx >> 6] &= ~(1ULL << mod64(idx));
  }

  // todo: make non branching
  xss_always_inline void set(const uint64_t idx, bool val) {
    if (val)
      set_one(idx);
    else
      set_zero(idx);
  }

  xss_always_inline bool get(const uint64_t idx) const {
    return data_[idx >> 6] & (1ULL << mod64(idx));
  }

  uint64_t size() const {
    return n_bits_;
  }

  uint64_t* data() {
    return data_;
  }

  const uint64_t* data() const {
    return data_;
  }

  ~bit_vector() {
    delete delete_data_;
  }

  bit_vector& operator=(bit_vector&& other) {
    n_bits_ = other.n_bits_;
    n_words_ = other.n_words_;
    n_bytes_ = other.n_bytes_;
    std::swap(data_, other.data_);
    return (*this);
  }

  bit_vector(bit_vector&& other) {
    (*this) = std::move(other);
  }
  bit_vector& operator=(const bit_vector& other) = delete;
  bit_vector(const bit_vector& other) = delete;
};

class parentheses_stream {
private:
  bit_vector& bv_;
  uint64_t* bv_data_;

  uint64_t current_word_macro_idx_;
  uint64_t current_word_micro_idx_;
  uint64_t current_word_;

  xss_always_inline void automatic_new_word() {
    if (xss_unlikely(++current_word_micro_idx_ == 64)) {
      bv_data_[current_word_macro_idx_++] = current_word_;
      current_word_micro_idx_ = 0;
      current_word_ = 0ULL;
    }
  }

public:
  parentheses_stream(bit_vector& bv)
      : bv_(bv),
        bv_data_(bv_.data()),
        current_word_macro_idx_(0),
        current_word_micro_idx_(0),
        current_word_(0ULL) {}

  xss_always_inline uint64_t bits_written() const {
    return (current_word_macro_idx_ << 6) + current_word_micro_idx_;
  }

  xss_always_inline void flush() {
    bv_data_[current_word_macro_idx_] = current_word_;
  }

  xss_always_inline void fetch() {
    current_word_ = bv_data_[current_word_macro_idx_];
  }

  xss_always_inline void append_opening_parenthesis() {
    current_word_ |= (1ULL << current_word_micro_idx_);
    automatic_new_word();
  }

  xss_always_inline void append_closing_parenthesis() {
    current_word_ &= ~(1ULL << current_word_micro_idx_);
    automatic_new_word();
  }

  xss_always_inline void append_copy(const uint64_t distance,
                                     const uint64_t length) {
    flush();
    const uint64_t rhs =
        (current_word_macro_idx_ << 6) + current_word_micro_idx_;
    const uint64_t lhs = rhs - distance;
    for (uint64_t i = 0; i < length; ++i) {
      bv_.set(rhs + i, bv_.get(lhs + i));
    }
    current_word_micro_idx_ += length;
    current_word_macro_idx_ += current_word_micro_idx_ >> 6;
    current_word_micro_idx_ =
        current_word_micro_idx_ - ((current_word_micro_idx_ >> 6) << 6);
    fetch();
  }

  ~parentheses_stream() {
    flush();
  }
};

} // namespace xss

namespace std {

[[maybe_unused]] static std::string to_string(const xss::bit_vector& bv) {
  std::stringstream result;
  for (uint64_t i = 0; i < bv.size(); ++i) {
    result << (bv.get(i) ? "(" : ")");
  }
  return result.str();
}

[[maybe_unused]] static std::ostream& operator<<(std::ostream& out,
                                                 const xss::bit_vector& bv) {
  return out << std::to_string(bv);
}

} // namespace std
