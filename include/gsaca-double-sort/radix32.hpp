#pragma once

#include <limits>
#include "uint_types.hpp"
#include "ips4o/ips4o.hpp"

namespace gsaca_lyndon {

namespace radix_internal {

// this seems to work best
constexpr size_t insertion_threshold = 32;

template<typename data_type>
using key_type = typename std::remove_reference<
    decltype(std::declval<data_type>().key)>::type;

template<typename data_type>
constexpr static size_t key_size = sizeof(key_type<data_type>);

template<typename data_type>
constexpr static size_t key_max =
    std::numeric_limits<key_type<data_type>>::max();

template<bool less, typename key_type>
bool compare(key_type const &a, key_type const &b) {
  if constexpr (less) return a < b;
  else return a > b;
}

// INSERTION SORT ==============================================================
template<bool increasing, typename data_type, typename count_type>
static inline void insertion(data_type *const data, count_type const n) {
  using key_type = key_type<data_type>;
  static_assert(std::is_unsigned_v<key_type>);
  
  // luxury: we have space to the left!  
  #if defined(__GNUC__) || defined(__clang__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Warray-bounds"
  #endif
  key_type h = data[-1].key;
  if constexpr (increasing)
    data[-1].key = 0;
  else
    data[-1].key = std::numeric_limits<key_type>::max();  
  #if defined(__GNUC__) || defined(__clang__)
  #pragma GCC diagnostic pop
  #endif


  for (count_type i = 1; i < n; ++i) {
    data_type const insert = data[i];
    int64_t insertion_index = i;
    while (compare<!increasing>(data[insertion_index - 1].key, insert.key)) {
      data[insertion_index] = data[insertion_index - 1];
      --insertion_index;
    }
    data[insertion_index] = insert;
  }

  #if defined(__GNUC__) || defined(__clang__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Warray-bounds"
  #endif
  data[-1].key = h;
  #if defined(__GNUC__) || defined(__clang__)
  #pragma GCC diagnostic pop
  #endif
}


// MSD RADIX SORT ==============================================================
template<bool increasing, size_t byte, typename data_type, typename count_type>
static inline void msd_radix_internal(data_type *const data,
                                      data_type *const buffer,
                                      count_type const n) {
  constexpr uint8_t key_shift = byte * 8;

  if (n < insertion_threshold) {
    insertion<increasing>(data, n);
    if constexpr (byte % 2 == 0) {
      for (count_type i = 0; i < n; ++i) {
        buffer[i] = data[i];
      }
    }
  } else {
    count_type histogram[256] = {};
    for (count_type i = 0; i < n; ++i) {
      ++histogram[(data[i].key >> key_shift) & 0xFF];
    }

    constexpr int start_bucket = increasing ? 0 : 255;
    constexpr int stop_bucket = increasing ? 256 : -1;
    constexpr int inc = increasing ? 1 : -1;
    count_type l = 0;
    for (int i = start_bucket; i != stop_bucket; i += inc) {
      count_type const bucket_size = histogram[i];
      histogram[i] = l;
      l += bucket_size;
    }

    for (count_type i = 0; i < n; ++i) {
      buffer[histogram[(data[i].key >> key_shift) & 0xFF]++] = data[i];
    }

    // no need to copy data back from buffer!
    if constexpr (byte != 0) {
      l = 0;
      for (int i = start_bucket; i != stop_bucket; i += inc) {
        count_type const bucket_size = histogram[i] - l;
        if (bucket_size > 0) {
          msd_radix_internal<increasing, byte - 1>(&(buffer[l]), &(data[l]),
                                                   bucket_size);
          l += bucket_size;
        }
      }
    }
  }
}


template<bool increasing, int key_bytes_tmp = -1,
    typename data_type, typename count_type>
static inline void msd_radix_internal(data_type *const data,
                                      data_type *const buffer,
                                      count_type const n,
                                      uint8_t const key_bytes) {
  if constexpr (radix_internal::key_size<data_type> > 4) {
    if (key_bytes == 5) {
      msd_radix_internal<increasing, 4>(data, buffer, n);
      for (count_type i = 0; i < n; ++i) {
        data[i] = buffer[i];
      }
      return;
    }
  }

  if (key_bytes == 4) {
    msd_radix_internal<increasing, 3>(data, buffer, n);
  }
  else if (key_bytes == 3) {
    msd_radix_internal<increasing, 2>(data, buffer, n);
    for (count_type i = 0; i < n; ++i) {
      data[i] = buffer[i];
    }
  }
  else if (key_bytes == 2) {
    msd_radix_internal<increasing, 1>(data, buffer, n);
  }
  else {
    msd_radix_internal<increasing, 0>(data, buffer, n);
    for (count_type i = 0; i < n; ++i) {
      data[i] = buffer[i];
    }
  }
}

// LSD RADIX SORT ==============================================================
template<bool increasing, size_t byte = 0,
    typename data_type, typename count_type>
static inline void
lsd_radix_internal(data_type *data, data_type *buffer, count_type const n,
                   uint8_t const key_bytes) {
  if constexpr (byte == 0) {
    if (key_bytes % 2) {
      for (count_type i = 0; i < n; ++i) {
        buffer[i] = data[i];
      }
      std::swap(data, buffer);
    }
  }

  constexpr uint8_t key_shift = byte * 8;
  count_type histogram[256] = {};
  for (count_type i = 0; i < n; ++i) {
    ++histogram[(data[i].key >> key_shift) & 0xFF];
  }

  constexpr int start_bucket = increasing ? 0 : 255;
  constexpr int stop_bucket = increasing ? 256 : -1;
  constexpr int inc = increasing ? 1 : -1;
  count_type l = 0;
  for (int i = start_bucket; i != stop_bucket; i += inc) {
    count_type const bucket_size = histogram[i];
    histogram[i] = l;
    l += bucket_size;
  }

  for (count_type i = 0; i < n; ++i) {
    buffer[histogram[(data[i].key >> key_shift) & 0xFF]++] = data[i];
  }

  if constexpr (byte + 1 < key_size<data_type>) {
    if (byte + 1 < key_bytes) {
      lsd_radix_internal<increasing, byte + 1>(buffer, data, n, key_bytes);
    }
  }
}

}


template<bool increasing = true, typename data_type>
static inline void
lsd_radix(data_type *const data, data_type *const buffer, size_t const n,
          size_t const max_key) {
  uint8_t const key_bytes = (71 - __builtin_clzl(max_key)) >> 3;
  radix_internal::lsd_radix_internal<increasing>(data, buffer, n, key_bytes);
}

template<bool increasing = true, typename data_type>
static inline void
msd_radix(data_type *const data, data_type *const buffer, size_t const n,
          size_t const max_key) {
  uint8_t const key_bytes = (71 - __builtin_clzl(max_key)) >> 3;
  radix_internal::msd_radix_internal<increasing>(data, buffer, n, key_bytes);
}

struct MSD {
  template<bool increasing = true, bool stable = true, typename data_type>
  static inline void
  sort(data_type *const data, data_type *const buffer, size_t const n,
       size_t const max_key = radix_internal::key_max<data_type>) {
    msd_radix<increasing>(data, buffer, n, max_key);
  }

  static std::string id() {
    return "msd";
  }
};

struct LSD {
  template<bool increasing = true, bool stable = true, typename data_type>
  static inline void
  sort(data_type *const data, data_type *const buffer, size_t const n,
       size_t const max_key = radix_internal::key_max<data_type>) {
    lsd_radix<increasing>(data, buffer, n, max_key);
  }

  static std::string id() {
    return "lsd";
  }
};

struct IPS4O {
  template<bool increasing = true, bool stable = false, typename data_type>
  static inline void
  sort(data_type *const data, data_type *const, size_t const n,
       size_t const = 0) {
    auto compare = [&](data_type const &a, data_type const &b) {
        if constexpr (increasing) {
          if constexpr (stable) {
            return a.key < b.key || (a.key == b.key && a.value < b.value);
          } else {
            return a.key < b.key;
          }
        } else {
          if constexpr (stable) {
            return a.key > b.key || (a.key == b.key && a.value < b.value);
          } else {
            return a.key > b.key;
          }
        }
    };

    ips4o::sort(data, data + n, compare);
  }

  static std::string id() {
    return "ips4o";
  }
};

template<typename key_type, typename value_type = key_type>
struct radix_key_val_pair {
  key_type key;
  value_type value;
};

} // namespace gsaca_lyndon
