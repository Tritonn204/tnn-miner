#pragma once

#include "phase_1.hpp"
#include "robin-hood.h"

namespace gsaca_lyndon {

template<typename p1_sorter = MSD, typename p2_sorter = MSD,
    typename buffer_type = auto_buffer_type,
    bool use_flags = true,
    typename index_type, // auto deduce
    typename value_type, // auto deduce
    typename used_buffer_type = get_buffer_type<buffer_type, index_type>>
static void gsaca_hash_ds(value_type const *const text, index_type *const sa,
                          size_t const n) {
  static_assert(std::is_unsigned<value_type>::value);
  static_assert(std::is_unsigned<index_type>::value);
  static_assert(std::is_unsigned<used_buffer_type>::value);
  static_assert(check_buffer_type<buffer_type, index_type, used_buffer_type>);
  //static_assert(sizeof(value_type) == 1);
  static_assert(sizeof(used_buffer_type) >= 4);

  using count_type = get_count_type<used_buffer_type, index_type>;
  using unordered_map64 = robin_hood::unordered_flat_map<uint64_t, used_buffer_type>;
  using p1_group_type = phase_1_group_type<used_buffer_type>;
  using p1_stack_type = phase_1_stack_type<used_buffer_type>;
  using F = flag_type<use_flags>;

  constexpr count_type MAX_HASHING = 8;

  auto lce = [&](count_type const i, count_type const j) {
      count_type lce = 0;
      while (text[i + lce] == text[j + lce]) ++lce;
      return lce;
  };

  // determine longest lyndon word at i
  auto naive_lyndon = [&](count_type const i) {
      count_type j = i + 1;
      count_type k = i;
      while (text[k] <= text[j]) {
        k = (text[k] < text[j]) ? i : k + 1;
        ++j;
      }
      return j - k;
  };

  count_type period_i = 0;
  count_type repetitions_i = 0;
  // determine longest lyndon word (or run, if word longer than max hashing)
  auto smart_lyndon = [&](count_type const i) {
      repetitions_i = 0;
      count_type const end = i + MAX_HASHING;
      count_type j = i + 1;
      count_type k = i;
      while (j < end && text[k] <= text[j]) {
        k = (text[k] < text[j]) ? i : k + 1;
        ++j;
      }
      if (j < end) {
        return j - k;
      } else {
        if (k == i) {
          return MAX_HASHING;
        } else {
          period_i = j - k;
          count_type const lce_kj = lce(k, j);
          count_type const runlen = MAX_HASHING + lce(k, j);
          repetitions_i = runlen / period_i;
          return (text[k + lce_kj] < text[j + lce_kj]) ? runlen : period_i;
        }
      }
  };

  struct sorted_group {
    used_buffer_type border;
    used_buffer_type size;
    used_buffer_type lyndon;
  };

  struct nano_text_nopad {
    uint64_t text;
    used_buffer_type first;

    gsaca_always_inline count_type lyndon() const {
      return 8 - (__builtin_ctzl(text) >> 3);
    }
  } __attribute((packed));

  constexpr int64_t pad = sizeof(sorted_group) - sizeof(used_buffer_type) - 8;
  struct nano_text_pad {
    uint64_t text;
    used_buffer_type first;
    uint8_t dummy[std::max((int64_t) 1, pad)] = {};

    gsaca_always_inline count_type lyndon() const {
      return 8 - (__builtin_ctzl(text) >> 3);
    }
  } __attribute((packed));

  using nano_text = typename std::conditional<(pad > 0),
      nano_text_pad, nano_text_nopad>::type;


  index_type *const nano_id_of = sa;
  static_assert(sizeof(sorted_group) == sizeof(nano_text));

  std::vector<nano_text> to_sort_nano;
  to_sort_nano.emplace_back(
      nano_text{0, (used_buffer_type) n - 1}); // group of n - 1
  to_sort_nano.emplace_back(nano_text{0x100, 0}); // group of 0
  {
    std::vector<used_buffer_type> first_occ_lookup16(std::pow(2, 16));
    unordered_map64 first_occ_lookup64;


    for (count_type i = 1; i < n - 1; ++i) {
      count_type const lyndon_i = smart_lyndon(i);
      if (gsaca_likely(repetitions_i < 3)) {
        // not a run, proceed normally
        if (lyndon_i == 1) {
          used_buffer_type &first = first_occ_lookup16[text[i]];
          if (gsaca_unlikely(first == 0)) {
            first = i;
            to_sort_nano.emplace_back(nano_text{((uint64_t) text[i]) << 56, i});
          } else {
            nano_id_of[i] = first;
          }
        } else if (lyndon_i == 2) {
          uint64_t const nano = ((uint64_t) text[i]) << 8 | text[i + 1];
          used_buffer_type &first = first_occ_lookup16[nano];
          if (gsaca_unlikely(first == 0)) {
            first = i;
            to_sort_nano.emplace_back(nano_text{nano << 48, i});
          } else {
            nano_id_of[i] = first;
            nano_id_of[++i] = first + 1;
          }
        } else {
          auto const lyn = std::min(lyndon_i, MAX_HASHING);
          uint64_t nano = text[i];
          for (count_type j = 1; j < lyn; ++j) {
            nano <<= 8;
            nano |= text[i + j];
          }
          used_buffer_type &first = first_occ_lookup64[nano];
          if (gsaca_unlikely(first == 0)) {
            first = i;
            to_sort_nano.emplace_back(nano_text{nano << ((8 - lyn) << 3), i});
          } else {
            nano_id_of[i] = first;
            if (lyn < MAX_HASHING) {
              for (count_type j = 1; j < lyn; ++j) {
                nano_id_of[i + j] = first + j;
              }
              i = i + lyn - 1;
            }
          }
        }
      } else {
        uint64_t nano = text[i];
        for (count_type j = 1; j < MAX_HASHING; ++j) {
          nano <<= 8;
          nano |= text[i + j];
        }
        used_buffer_type &first = first_occ_lookup64[nano];
        if (gsaca_unlikely(first == 0)) {
          first = i;
          to_sort_nano.emplace_back(nano_text{nano, i});
        } else {
          nano_id_of[i] = first;
        }

        count_type const period = period_i;
        count_type const repetitions = repetitions_i;
        count_type const stop = i + period;
        for (++i; i < stop; ++i) {
          auto const lyn = naive_lyndon(i);
          if (lyn == 1) {
            used_buffer_type &first = first_occ_lookup16[text[i]];
            if (gsaca_unlikely(first == 0)) {
              first = i;
              to_sort_nano.emplace_back(
                  nano_text{((uint64_t) text[i]) << 56, i});
            } else {
              nano_id_of[i] = first;
            }
          } else if (lyn == 2) {
            uint64_t const nano = ((uint64_t) text[i]) << 8 | text[i + 1];
            used_buffer_type &first = first_occ_lookup16[nano];
            if (gsaca_unlikely(first == 0)) {
              first = i;
              to_sort_nano.emplace_back(nano_text{nano << 48, i});
            } else {
              nano_id_of[i] = first;
              nano_id_of[++i] = first + 1;
            }
          } else {
            uint64_t nano = text[i];
            for (count_type j = 1; j < lyn; ++j) {
              nano <<= 8;
              nano |= text[i + j];
            }
            used_buffer_type &first = first_occ_lookup64[nano];
            if (gsaca_unlikely(first == 0)) {
              first = i;
              to_sort_nano.emplace_back(nano_text{nano << ((8 - lyn) << 3), i});
            } else {
              nano_id_of[i] = first;
              for (count_type j = 1; j < lyn; ++j) {
                nano_id_of[i + j] = first + j;
              }
              i = i + lyn - 1;
            }
          }
        }

        // now we have i = original i + period
        count_type const last_copy =
            i + (repetitions - 2) * period - MAX_HASHING;

        for (; i < last_copy; ++i) {
          nano_id_of[i] = i - period;
        }
        --i;
      }
    }
  }
  nano_id_of[n - 1] = 0;
  nano_id_of[0] = 1;


  count_type const initial_group_count = to_sort_nano.size();
  std::sort(to_sort_nano.begin(), to_sort_nano.end(),
            [](nano_text const &a, nano_text const &b) {
                return a.text < b.text;
            });


  for (count_type g = 2; g < initial_group_count; ++g) {
    nano_id_of[to_sort_nano[g].first] = -g;
  }
  for (count_type i = 1; i < n - 1; ++i) {
    nano_id_of[i] = (gsaca_likely(nano_id_of[i] < n))
                    ? nano_id_of[nano_id_of[i]]
                    : ((index_type) -nano_id_of[i]);
  }

  // warning can be ignored because equal size of nano_text and
  // sorted_group has been asserted!
  #if defined(__GNUC__) || defined(__clang__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Waddress-of-packed-member"
  #endif
  sorted_group *const sorted_groups = (sorted_group *) to_sort_nano.data();
  #if defined(__GNUC__) || defined(__clang__)
  #pragma GCC diagnostic pop
  #endif

  for (count_type g = 2; g < initial_group_count; ++g) {
    sorted_groups[g].lyndon = to_sort_nano[g].lyndon();
    sorted_groups[g].size = 0;
  }

  for (count_type i = 1; i < n - 1; ++i) {
    ++(sorted_groups[nano_id_of[i]].size);
  }

  count_type border = 2;
  for (count_type g = 2; g < initial_group_count; ++g) {
    sorted_groups[g].border = border;
    border += sorted_groups[g].size;
  }

  used_buffer_type *const isa = (used_buffer_type *) malloc(
      n * sizeof(used_buffer_type));

  // sort the SA
  for (count_type j = 1; j < n - 1; ++j) {
    isa[j] = nano_id_of[j];
  }
  for (count_type j = 1; j < n - 1; ++j) {
    isa[j] = sorted_groups[isa[j]].border++;
  }
  // add flags now, while still sequential text access
  for (count_type j = 1; j < n - 1; ++j) {
    isa[j] = F::conditional_add_flag(text[j - 1] < text[j], isa[j]);
  }
  for (count_type j = 1; j < n - 1; ++j) {
    sa[F::remove_flag(isa[j])] = F::conditional_add_flag(F::is_flagged(isa[j]),
                                                         j);
  }
  sa[0] = n - 1;
  sa[1] = 0;

  for (count_type g = initial_group_count - 1; g > 2; --g) {
    sorted_groups[g].border = sorted_groups[g - 1].border;
  }
  sorted_groups[2].border = 2;

  p1_stack_type p1_groups;
  for (count_type g = 2; g < initial_group_count; ++g) {
    if (gsaca_unlikely(sorted_groups[g].lyndon == MAX_HASHING)) {
      count_type const first = F::remove_flag(sa[sorted_groups[g].border]);
      count_type const target = first + MAX_HASHING;
      count_type next = first + 1;
      for (count_type j = next + 1; j < target; ++j) {
        if (text[j] < text[next]) {
          next = j;
        }
      }
      p1_groups.emplace_back(p1_group_type{sorted_groups[g].border,
                                           sorted_groups[g].size,
                                           next - first, true, false});
    } else {
      p1_groups.emplace_back(p1_group_type{sorted_groups[g].border,
                                           sorted_groups[g].size,
                                           sorted_groups[g].lyndon,
                                           false, true});
    }
  }
  { auto remove = std::move(to_sort_nano); }



  auto p2_input_groups = phase_1_by_sorting<p1_sorter, F>(sa, isa, p1_groups);

  phase_2_by_sorting<p2_sorter, F>(sa, isa, n, p2_input_groups.data(),
                                   (index_type) p2_input_groups.size());
  free(isa);
}

}
