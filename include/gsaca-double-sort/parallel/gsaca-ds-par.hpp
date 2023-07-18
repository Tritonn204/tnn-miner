#pragma once

#include <omp.h>
#include "../extract.hpp"
#include "phase_1.hpp"
#include "phase_2.hpp"

namespace gsaca_lyndon {

namespace double_sort_internal {

template<typename buffer_type, typename F,
         typename index_type, typename value_type>
auto sort_by_prefix_parallel(value_type const *const text, index_type *const sa,
                    get_count_type <index_type, buffer_type> const n, uint8_t const prefix, size_t const threads) {
  using count_type = get_count_type<index_type, buffer_type>;
  using p1_stack_type = phase_1_stack_type<buffer_type>;
  using p1_group_type = typename p1_stack_type::value_type;
  p1_stack_type result;

  if (sizeof(value_type) == 1) {
    if (prefix == 1) {
        std::vector<count_type> histogram_vec(256*threads);
        count_type* const histogram_cont = histogram_vec.data();

		// counting
		#pragma omp parallel for
		for (size_t i = 0; i < threads; ++i) {
		   count_type interval_begin = std::max((count_type) (i * (n / threads + (n % threads > 0))), (count_type)1);
		   count_type interval_end = std::min((count_type) ((i + 1) * (n / threads + (n % threads > 0))), n-1);
		   count_type* histogram = &(histogram_cont[256*i]);

		   for (count_type j = interval_begin; j < interval_end; ++j) {
		       ++histogram[text[j]];
		   }
    	}

		// calculate borders
		count_type border = 2;
		for (index_type i = 1; i < 256; ++i) {
		    count_type gsize = 0;
		    for (size_t j = 0; j < threads; ++j) {
		        count_type bucket = 256*j+i;
		        count_type count = histogram_cont[bucket];
		        histogram_cont[bucket] = border;
		        border += count;
		        gsize += count;
		    }
		    if (gsize > 0 && i > 0) {
		      result.emplace_back(p1_group_type{border-gsize, gsize, 1, true, false});
		    }
		}

		// distribute
		#pragma omp parallel for
		for (size_t i = 0; i < threads; ++i) {
		    count_type interval_begin = i * (n / threads + (n % threads > 0));
		    count_type interval_end = std::min((count_type)((i + 1) * (n / threads + (n % threads > 0))), n);
		    count_type* borders = &(histogram_cont[256*i]);

		    for (count_type j = interval_begin; j < interval_end; ++j) {
		        sa[borders[text[j]]++] = j;
		    }
		}
  } else {
      count_type const buckets = 1ULL << (prefix << 3);
      std::vector<count_type> histogram_vec(buckets*threads);
      count_type* const histogram_cont = histogram_vec.data();
      count_type const stop = n - prefix - 1;

      // counting
      #pragma omp parallel for
      for (size_t i = 0; i < threads; ++i) {
          count_type interval_begin = std::max((count_type) (i * (n / threads + (n % threads > 0))), (count_type)1);
          count_type interval_end = std::min((count_type) ((i + 1) * (n / threads + (n % threads > 0))), stop);
          count_type* histogram = &(histogram_cont[buckets*i]);

          for (count_type j = interval_begin; j < interval_end; ++j) {
              ++histogram[extract(text, j, prefix)];
          }
      }
      {
          count_type* histogram = &(histogram_cont[buckets*(threads-1)]);
          for (count_type i = stop; i < n - 1; ++i) {
            ++histogram[safe_extract(text, i, prefix)];
          }
      }

      // calculate borders
      count_type border = 2;
      for (count_type i = buckets >> 8; i < buckets; ++i) {
          count_type gsize = 0;
          for (size_t j = 0; j < threads; ++j) {
              count_type bucket = buckets*j+i;
              count_type count = histogram_cont[bucket];
              histogram_cont[bucket] = border;
              border += count;
              gsize += count;
          }
          if (gsize > 0 && i > 0) {
            result.emplace_back(p1_group_type{border-gsize, gsize, 1, true, false});
          }
      }

      // distribute
      #pragma omp parallel for
      for (size_t i = 0; i < threads; ++i) {
          count_type interval_begin = std::max(i * (n / threads + (n % threads > 0)), (size_t)1);
          count_type interval_end = std::min((count_type)((i + 1) * (n / threads + (n % threads > 0))), stop);
          count_type* borders = &(histogram_cont[buckets*i]);

          for (count_type j = interval_begin; j < interval_end; ++j) {
              sa[borders[extract(text, j, prefix)]++] = F::conditional_add_flag(
                          text[j - 1] < text[j], j);
          }
      }
      {
          count_type* borders = &(histogram_cont[buckets*(threads-1)]);
          for (count_type i = stop; i < n - 1; ++i) {
              sa[borders[safe_extract(text, i, prefix)]++] = F::conditional_add_flag(
                          text[i - 1] < text[i], i);
          }
      }
  }
}
  else {
      // fill sa with values
      #pragma omp parallel for
      for (count_type i = 0; i < n; ++i) {
          sa[i] = i;
      }

      // sort sa by first character
      auto comp = [&](auto a, auto b) {
           auto extracted1 = safe_extract(text, a, prefix);
           auto extracted2 = safe_extract(text, b, prefix);
           return (extracted1 < extracted2) || ((extracted1 == extracted2) && a < b);
      };
      ips4o::parallel::sort(&(sa[0]), &(sa[n]), comp);

      // determine gsizes
      // TODO: Parallelize
      count_type left_border = 2;
      count_type gsize = 1;
      for (count_type i = 2; i < n-1; ++i) {
          if (safe_extract(text, sa[i], prefix) == safe_extract(text, sa[i+1], prefix)) {
              ++gsize;
          }
          else {
              result.emplace_back(p1_group_type{left_border, gsize, 1, true, false});
              left_border = i+1;
              gsize = 1;
          }
      }
      result.emplace_back(p1_group_type{left_border, gsize, 1, true, false});

      // add flags
      #pragma omp parallel for
      for (count_type i = 0; i < n; ++i) {
          auto idx = sa[i];
          sa[i] = (idx != 0) ? F::conditional_add_flag(text[idx - 1] < text[idx], idx) : idx;
      }
  }
  sa[0] = n - 1;
  sa[1] = 0;
  return result;
}

}

template<typename buffer_type = auto_buffer_type,
    bool use_flags = true,
    typename index_type, // auto deduce
    typename value_type, // auto deduce
    typename used_buffer_type = get_buffer_type <buffer_type, index_type>>
static void
gsaca_ds_par(value_type const *const text, index_type *const sa, size_t const n, size_t const threads,
         size_t const initial_sort_prefix_len = 1) {
  static_assert(std::is_unsigned<value_type>::value);
  static_assert(std::is_unsigned<index_type>::value);
  static_assert(std::is_unsigned<used_buffer_type>::value);
  static_assert(check_buffer_type<buffer_type, index_type, used_buffer_type>);
  //static_assert(sizeof(value_type) == 1);

  using F = flag_type<use_flags>;

  size_t const p_max = omp_get_max_threads();
  size_t const p = (threads == 0) ? p_max : threads;
  omp_set_dynamic(0);
  omp_set_num_threads(p);

  auto p1_input_groups =
      double_sort_internal::sort_by_prefix_parallel<used_buffer_type, F>
            (text, sa, n, initial_sort_prefix_len, p);
  used_buffer_type *const isa = (used_buffer_type *) malloc(n * sizeof(used_buffer_type));

  auto p2_input_groups = phase_1_by_sorting_parallel<F>(sa, isa, p1_input_groups, p);
  
  phase_2_by_sorting_stable_parallel<F>(sa, isa, n, p2_input_groups.data(),
                     p2_input_groups.size(), p);
  free(isa);

  omp_set_num_threads(p_max);
}

template<typename buffer_type = auto_buffer_type,
    typename index_type, // auto deduce
    typename value_type>
static void gsaca_ds1_par(value_type const *const text, index_type *const sa,
                      size_t const n, size_t const threads) {
  gsaca_ds_par<buffer_type, false>(text, sa, n, threads, 1);
}

template<typename buffer_type = auto_buffer_type,
    typename index_type, // auto deduce
    typename value_type>
static void gsaca_ds2_par(value_type const *const text, index_type *const sa,
                      size_t const n, size_t const threads) {
  gsaca_ds_par<buffer_type>(text, sa, n, threads, 2);
}

template<typename buffer_type = auto_buffer_type,
    typename index_type, // auto deduce
    typename value_type>
static void gsaca_ds3_par(value_type const *const text, index_type *const sa,
                      size_t const n, size_t const threads) {
  gsaca_ds_par<buffer_type>(text, sa, n, threads, 3);
}

}
