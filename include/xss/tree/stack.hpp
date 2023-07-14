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
#include <cmath>
#include <stack>

class telescope_stack {
private:
  std::stack<uint64_t> data_left_;
  std::stack<uint64_t> data_right_;

  uint64_t top_bit_;
  uint64_t top_bit_mod64_;
  uint64_t top_value_;

  uint64_t top_word_;

public:
  telescope_stack()
      : top_bit_(0), top_bit_mod64_(0), top_value_(0), top_word_(1ULL) {
    // fill last word with 1s
    data_right_.push(std::numeric_limits<uint64_t>::max());
  }

  telescope_stack(const uint64_t) : telescope_stack() {}

  xss_always_inline void push(const uint64_t value) {
    uint64_t offset = value - top_value_;
    if (xss_unlikely(offset > 127)) {
      data_right_.push(top_value_);
      data_right_.push(top_bit_);
    } else {
      top_bit_ += offset;
      top_bit_mod64_ += offset;
      while (top_bit_mod64_ > 63) {
        top_bit_mod64_ -= 64;
        data_left_.push(top_word_);
        top_word_ = 0ULL;
      }
      top_word_ |= (1ULL << top_bit_mod64_);
    }
    top_value_ = value;
  }

  xss_always_inline uint64_t top() const {
    return top_value_;
  }

  xss_always_inline void pop() {
    if (xss_unlikely(top_bit_ == data_right_.top())) {
      data_right_.pop();
      top_value_ = data_right_.top();
      data_right_.pop();
    } else {
      const uint64_t previous_top_bit_ = top_bit_;
      top_word_ &= ~(1ULL << top_bit_mod64_);
      while (top_word_ == 0ULL) {
        top_word_ = data_left_.top();
        data_left_.pop();
      }
      top_bit_mod64_ = 63 - __builtin_clzl(top_word_);
      top_bit_ = ((data_left_.size()) << 6) + top_bit_mod64_;
      top_value_ -= previous_top_bit_ - top_bit_;
    }
  }

  telescope_stack& operator=(telescope_stack&& other) {
    top_bit_ = other.top_bit_;
    top_bit_mod64_ = other.top_bit_mod64_;
    top_value_ = other.top_value_;
    top_word_ = other.top_word_;
    std::swap(data_left_, other.data_left_);
    std::swap(data_right_, other.data_right_);
    return (*this);
  }

  telescope_stack(telescope_stack&& other) {
    (*this) = std::move(other);
  }

  telescope_stack(const telescope_stack&) = delete;
  telescope_stack& operator=(const telescope_stack&) = delete;
};

class reverse_telescope_stack {
private:
  telescope_stack base_stack;

public:
  constexpr static uint64_t max_val = std::numeric_limits<uint64_t>::max();

  xss_always_inline uint64_t top() const {
    return max_val - base_stack.top();
  }

  xss_always_inline void pop() {
    base_stack.pop();
  }

  xss_always_inline void push(const uint64_t e) {
    base_stack.push(max_val - e);
  }
};

// non-shrinking buffer (since we are interested in the memory peak only)
template <typename stack_type, typename index_type>
class buffered_stack {
private:
  const uint64_t buffer_capacity_;
  const uint64_t buffer_half_capacity_;

  index_type cur_buffer_capacity_;
  index_type cur_buffer_size_;
  index_type* cur_buffer_;

  stack_type base_stack_;

  xss_always_inline uint64_t get_buffer_capacity(const uint64_t buffer_bytes) {
    const uint64_t actual_bytes =
        std::max(buffer_bytes, (uint64_t)(64ULL * 1024));
    const uint64_t log_bytes = (uint64_t) std::floor(std::log2(actual_bytes));
    return (1ULL << log_bytes) / sizeof(index_type);
  }

public:
  buffered_stack(const uint64_t buffer_bytes, stack_type&& stack)
      : buffer_capacity_(get_buffer_capacity(buffer_bytes)),
        buffer_half_capacity_(buffer_capacity_ >> 1),
        cur_buffer_capacity_(64ULL * 1024 / sizeof(index_type)),
        cur_buffer_size_(1),
        base_stack_(std::move(stack)) {
    static_assert(sizeof(index_type) == 4 || sizeof(index_type) == 8);
    cur_buffer_ =
        (index_type*) malloc(cur_buffer_capacity_ * sizeof(index_type));
    cur_buffer_[0] = 0ULL;
  }

  xss_always_inline uint64_t top() const {
    return cur_buffer_[cur_buffer_size_ - 1];
  }

  xss_always_inline void push(const uint64_t e) {
    // check if buffer is full
    if (xss_unlikely(cur_buffer_size_ == cur_buffer_capacity_)) {
      if (xss_unlikely(cur_buffer_capacity_ == buffer_half_capacity_)) {
        for (uint64_t i = 0; i < cur_buffer_capacity_; ++i) {
          base_stack_.push(cur_buffer_[i] + 1);
        }
        delete cur_buffer_;
        cur_buffer_ =
            (index_type*) malloc(buffer_capacity_ * sizeof(index_type));
        cur_buffer_capacity_ = buffer_capacity_;
        cur_buffer_size_ = 1;
        cur_buffer_[0] = e;
      } else if (cur_buffer_capacity_ == buffer_capacity_) {
        for (uint64_t i = 0; i < buffer_half_capacity_; ++i) {
          base_stack_.push(cur_buffer_[i] + 1);
          cur_buffer_[i] = cur_buffer_[i + buffer_half_capacity_];
        }
        cur_buffer_size_ = buffer_half_capacity_ + 1;
        cur_buffer_[buffer_half_capacity_] = e;
      } else {
        index_type* new_buffer = (index_type*) malloc(
            (cur_buffer_capacity_ << 1) * sizeof(index_type));
        for (uint64_t i = 0; i < cur_buffer_capacity_; ++i) {
          new_buffer[i] = cur_buffer_[i];
        }
        new_buffer[cur_buffer_capacity_] = e;
        delete cur_buffer_;
        cur_buffer_ = new_buffer;
        cur_buffer_capacity_ <<= 1;
        ++cur_buffer_size_;
      }
      return;
    }

    cur_buffer_[cur_buffer_size_++] = e;
  }

  xss_always_inline void pop() {
    --cur_buffer_size_;
    if (xss_unlikely(cur_buffer_size_ == 0)) {
      cur_buffer_size_ = cur_buffer_capacity_ >> 1;
      for (uint64_t i = cur_buffer_size_; i > 0;) {
        cur_buffer_[--i] = base_stack_.top() - 1;
        base_stack_.pop();
      }
      return;
    }
  }

  ~buffered_stack() {
    delete cur_buffer_;
  }

  buffered_stack& operator=(buffered_stack&& other) = delete;
  buffered_stack(buffered_stack&& other) = delete;
  buffered_stack(const buffered_stack&) = delete;
  buffered_stack& operator=(const buffered_stack&) = delete;
};