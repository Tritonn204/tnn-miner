#pragma once

#include <omp.h>
#include "../uint_types.hpp"
#include "../radix32.hpp"

namespace gsaca_lyndon {
    
const size_t seq_threshold = 1025;

template<typename F = flag_type<false>, typename index_type, typename buffer_type>
inline void phase_2_by_sorting_stable_parallel(index_type *const sa, buffer_type *const isa, size_t const n,
                               phase_2_group_type<buffer_type> const *const groups,
                               size_t const number_of_groups, size_t threads) {
  using count_type = get_count_type<index_type, buffer_type>;
  using key_value_pair = radix_key_val_pair<buffer_type>;

  count_type max_group_size = 0;
  #pragma omp parallel for reduction(max:max_group_size)
  for (count_type g = 0; g < number_of_groups; ++g) {
    max_group_size = std::max(max_group_size, (count_type) groups[g].size);
  }

  constexpr count_type sg_count_threshold = 256ULL * 1024; // 1MiB buffer
  void *memory = malloc(
      sg_count_threshold * sizeof(count_type) +
      ((max_group_size + 1) << 1) * sizeof(key_value_pair));

  count_type *const subgroup_border_buffer = (count_type *) memory;
  key_value_pair *grouped_indices = (key_value_pair *) (subgroup_border_buffer +
                                                        sg_count_threshold);
  key_value_pair *grouped_indices_buffer = grouped_indices + max_group_size + 1;

  buffer_type *const subgroup_size =
      (buffer_type *) &(grouped_indices_buffer[0]);
  buffer_type *const subgroup_id =
      (buffer_type *) &(grouped_indices_buffer[(max_group_size >> 1) + 1]);

  count_type left_border = 2;

  for (count_type g = 2; g < number_of_groups; ++g) {
    count_type const gsize = groups[g].size;

    if (gsize == 1) {
      sa[left_border] = F::remove_flag(sa[left_border]);
      isa[sa[left_border]] = left_border;
      ++left_border;
    }
    else if (gsize < seq_threshold) {
      count_type const lyn = groups[g].lyndon;
      index_type *const sa_interval = &(sa[left_border]);

      for (count_type i = 0; i < gsize + 1; ++i) {
        subgroup_size[i] = 0;
      }

      subgroup_id[gsize - 1] = 0;
      subgroup_size[0] = 1;
      for (count_type i = gsize - 1; i > 0; --i) {
        subgroup_id[i - 1] = (F::remove_flag(sa_interval[i]) ==
                              (F::remove_flag(sa_interval[i - 1]) + lyn))
                             ? (subgroup_id[i] + 1)
                             : ((buffer_type) 0);
        ++subgroup_size[subgroup_id[i - 1]];
      }

      count_type sg_count = 0;
      while (subgroup_size[sg_count] > 0) ++sg_count;

      count_type *const subgroup_border =
          (gsaca_likely(sg_count < sg_count_threshold))
          ? (subgroup_border_buffer)
          : ((count_type *) malloc(sg_count * sizeof(count_type)));


      count_type local_left_border = 0;
      for (count_type i = 0; i < sg_count; ++i) {
        subgroup_border[i] = local_left_border;
        local_left_border += subgroup_size[i];
      }

      for (count_type i = 0; i < gsize; ++i) {
        count_type &border = subgroup_border[subgroup_id[i]];
        grouped_indices[border++].value = sa_interval[i];
      }

      count_type previous_border = 0;
      for (count_type j = 0; j < sg_count; ++j) {
        count_type const stop = subgroup_border[j];

        // retrieve lexicographical rank of inducers
        for (count_type i = previous_border; i < stop; ++i) {
          grouped_indices[i].key = isa[
              F::remove_flag(grouped_indices[i].value) + lyn];
        }

        if (gsaca_likely(stop - previous_border < 33)) {
          radix_internal::insertion<true>(
              &(grouped_indices[previous_border]), stop - previous_border);
        } else {
            gsaca_lyndon::msd_radix(&(grouped_indices[previous_border]),
                                  grouped_indices_buffer,
                                  stop - previous_border, n - 1);
        }

        for (count_type i = previous_border; i < stop; ++i) {
          sa_interval[i] = grouped_indices[i].value;
        }
        for (count_type i = previous_border; i < stop; ++i) {
          if (!F::is_flagged(sa_interval[i])) {
            isa[sa_interval[i]] = left_border + i;
          } else {
            sa_interval[i] = F::remove_flag(sa_interval[i]);
          }
        }
        previous_border = stop;
      }

      if (gsaca_unlikely(sg_count >= sg_count_threshold)) {
        free(subgroup_border);
      }

      left_border += gsize;
    }
    else {
      buffer_type const lyn = groups[g].lyndon;
      index_type *const sa_interval = &(sa[left_border]);

      // calculate subgroup_id and sg_count
      count_type* length_end = &(subgroup_border_buffer[0]);
      #pragma omp parallel for
      for (size_t i = 0; i < threads; ++i) {
          count_type interval_begin = i * (gsize / threads);
          count_type interval_end = i < threads-1 ? (count_type)((i + 1) * (gsize / threads)) : gsize;

          length_end[i] = 1;
          subgroup_id[interval_end-1] = 0;
          bool is_start = true;
          for (count_type j = interval_end-1; j > interval_begin; --j) {
              subgroup_id[j - 1] = (F::remove_flag(sa_interval[j]) == (F::remove_flag(sa_interval[j - 1]) + lyn))
                                   ? (subgroup_id[j] + 1)
                                   : ((buffer_type) 0);
              if ((is_start = (is_start && (subgroup_id[j - 1] > 0))))
                    ++length_end[i];
          }
      }
      for (size_t i = threads-2; i < threads-1; --i) {
          count_type interval_end = i < threads-1 ? (count_type)((i + 1) * (gsize / threads)) : gsize;
          count_type length = length_end[i];

          for (count_type j = interval_end; j > interval_end-length; --j) {
              subgroup_id[j - 1] = (F::remove_flag(sa_interval[j]) == (F::remove_flag(sa_interval[j - 1]) + lyn))
                                   ? (subgroup_id[j] + 1)
                                   : ((buffer_type) 0);
          }
      }
      count_type sg_count = 1;
      #pragma omp parallel for reduction(max:sg_count)
      for (count_type i = 0; i < gsize; ++i) {
          sg_count = std::max(sg_count, (count_type) subgroup_id[i]+1);
      }

      count_type *const subgroup_border =
          (gsaca_likely(threads*sg_count < sg_count_threshold))
          ? (subgroup_border_buffer)
          : ((count_type *) malloc(threads*sg_count * sizeof(count_type)));

      for (count_type i = 0; i < threads*sg_count; ++i)
          subgroup_border[i] = 0;

      // calculate subgroup_sizes
      #pragma omp parallel for
      for (size_t i = 0; i < threads; ++i) {
          count_type interval_begin = i * (gsize / threads);
          count_type interval_end = i < threads-1 ? (count_type)((i + 1) * (gsize / threads)) : gsize;
          count_type* subgroup_sizes = &(subgroup_border[i*sg_count]);

          for (count_type j = interval_begin; j < interval_end; ++j) {
              ++subgroup_sizes[subgroup_id[j]];
          }
      }

      // calculate subgroup_borders
      count_type sum = 0;
      for (count_type i = 0; i < sg_count; ++i) {
          for (size_t j = 0; j < threads; ++j) {
              count_type bucket_start = j*sg_count+i;
              count_type tmp = subgroup_border[bucket_start];
              subgroup_border[bucket_start] = sum;
              sum += tmp;
          }
      }

      // distribute
      #pragma omp parallel for
      for (size_t i = 0; i < threads; ++i) {
          count_type interval_begin = i * (gsize / threads);
          count_type interval_end = i < threads-1 ? (count_type)((i + 1) * (gsize / threads)) : gsize;
          count_type* subgroup_border_thread = &(subgroup_border[i*sg_count]);

          for (count_type j = interval_begin; j < interval_end; ++j) {
              count_type &border = subgroup_border_thread[subgroup_id[j]];
              grouped_indices[border++] = {0, sa_interval[j]};
          }
      }

      count_type previous_border = 0;
      for (count_type j = 0; j < sg_count; ++j) {
        count_type const stop = subgroup_border[(threads-1)*sg_count+j]; // last chunk contains end borders
        // retrieve lexicographical rank of inducers
        #pragma omp parallel for
        for (count_type i = previous_border; i < stop; ++i) {
          grouped_indices[i].key = isa[F::remove_flag(grouped_indices[i].value) + lyn];
        }

        auto comp = [&](auto a, auto b) {
           return a.key < b.key;
        };
        ips4o::parallel::sort(&(grouped_indices[previous_border]), &(grouped_indices[stop]), comp, threads);

        #pragma omp parallel for
        for (count_type i = previous_border; i < stop; ++i) {
          sa_interval[i] = grouped_indices[i].value;
        }
        #pragma omp parallel for
        for (count_type i = previous_border; i < stop; ++i) {
            if (!F::is_flagged(sa_interval[i])) {
              isa[sa_interval[i]] = left_border + i;
            } else {
              sa_interval[i] = F::remove_flag(sa_interval[i]);
            }
        }
        previous_border = stop;
      }

      if (gsaca_unlikely(sg_count >= sg_count_threshold)) {
        free(subgroup_border);
      }

      left_border += gsize;
    }
  }
}


} // namespace gsaca_lyndon
