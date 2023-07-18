#pragma once

#include <cstring>
#include <omp.h>
#include "../phase_types.hpp"
#include "phase_2.hpp"

namespace gsaca_lyndon {

template<typename F = flag_type<false>, typename index_type, typename buffer_type>
inline auto phase_1_by_sorting_parallel(index_type *const sa, buffer_type *const isa,
                               phase_1_stack_type<buffer_type> &input_groups, size_t threads,
                               size_t max_group_size = 0) {
  using count_type = get_count_type<index_type, buffer_type>;
  using output_type = phase_2_group_type<buffer_type>;
  using input_type = phase_1_group_type<buffer_type>;
  using sorting_type = radix_key_val_pair<buffer_type>;

  if (max_group_size == 0) {
    #pragma omp parallel for reduction(max:max_group_size)
    for (count_type i = 0; i < input_groups.size(); ++i) {
      max_group_size = std::max(max_group_size, (size_t) input_groups[i].size);
    }
  }

  count_type const n = input_groups.back().start + input_groups.back().size;

  // set isa to 0!
  buffer_type *const rank = isa;
  memset(rank, 0, n * sizeof(buffer_type));

  std::vector<output_type> result_groups(1);

  // twice the size for out-of-place radix sort
  sorting_type *to_sort = (sorting_type *) malloc(
      (max_group_size) * sizeof(sorting_type) * 2);

  buffer_type *const subgroup_id = (buffer_type *) to_sort;

  while (!input_groups.empty()) {
    auto const group = input_groups.back();
    input_groups.pop_back();

    count_type const gcontext = group.context;
    count_type const gsize = group.size;
    buffer_type const gstart = group.start;
    index_type *const sa_interval = &(sa[gstart]);

    if (gsize < seq_threshold) {
        if (gsize == 1) {
            index_type const idx = F::remove_flag(sa_interval[0]);
            rank[idx] = result_groups.size();
            count_type context = gcontext;
            while (rank[idx + context] != 0) {
              context += result_groups[rank[idx + context]].lyndon;
            }
            result_groups.emplace_back(output_type{context, 1});
          } else if (!group.check_for_runs) {
            // this group can directly be processed
            if (group.is_final) {
              // great! we can assign the rank!
              buffer_type const assign_rank = result_groups.size();
              for (count_type i = 0; i < gsize; ++i) {
                rank[F::remove_flag(sa_interval[i])] = assign_rank;
              }
              result_groups.emplace_back(output_type{gcontext, gsize});
            } else {
              // let's sort the group by the rank behind the context
              for (count_type i = 0; i < gsize; ++i) {
                to_sort[i].value = sa_interval[i];
              }
              for (count_type i = 0; i < gsize; ++i) {
                  //std::cout << "n: " << n << ", idx_before: " << to_sort[i].value << ", idx_after: " << F::remove_flag(to_sort[i].value) << ", gcontext: " << gcontext << std::endl;
                to_sort[i].key = rank[F::remove_flag(to_sort[i].value) + gcontext];
              }

              size_t max_rank = result_groups.size() - 1;
              msd_radix<false>(to_sort, to_sort + gsize, gsize, max_rank);

              for (count_type i = 0; i < gsize; ++i) {
                sa_interval[i] = to_sort[i].value;
              }

              count_type sg_size = 1;
              buffer_type sg_start = 0;
              buffer_type sg_key = to_sort[0].key;
              buffer_type sg_context = gcontext + result_groups[sg_key].lyndon;
              for (count_type i = 1; i < gsize; ++i) {
                if (to_sort[i].key == sg_key) {
                  ++sg_size;
                } else {
                  input_groups.emplace_back(input_type{gstart + sg_start,
                                                       sg_size,
                                                       sg_context,
                                                       true,
                                                       false});
                  sg_start = i;
                  sg_size = 1;
                  sg_key = to_sort[i].key;
                  sg_context = gcontext + result_groups[sg_key].lyndon;
                }
              }
              input_groups.emplace_back(
                  input_type{gstart + sg_start, sg_size, sg_context, true, false});
            }
          } else {
            buffer_type *const subgroup_size = subgroup_id + gsize;
            memset(subgroup_size, 0, (gsize + 2) * sizeof(buffer_type));
            subgroup_id[gsize - 1] = (rank[F::remove_flag(sa_interval[gsize - 1]) +
                                           gcontext]) ? (1)
                                                      : (0);
            subgroup_size[subgroup_id[gsize - 1]] = 1;
            for (count_type i = gsize - 1; i > 0; --i) {
              subgroup_id[i - 1] =
                  ((rank[F::remove_flag(sa_interval[i - 1]) + gcontext])
                   ? ((buffer_type) 1) :
                   (((F::remove_flag(sa_interval[i - 1]) + gcontext) !=
                     F::remove_flag(sa_interval[i]))
                    ? ((buffer_type) 0) :
                    ((subgroup_id[i]) ? (subgroup_id[i] + 1) : ((buffer_type) 0))));
              ++subgroup_size[subgroup_id[i - 1]];
            }

            count_type first_empty_subgroup = 1;
            while (subgroup_size[first_empty_subgroup] > 0) {
              ++first_empty_subgroup;
            }

            if (subgroup_size[0] > 0) {
              input_groups.emplace_back(
                  input_type{gstart, subgroup_size[0], gcontext, false, true});
            }

            count_type local_left_border = subgroup_size[0];
            subgroup_size[0] = 0;

            for (count_type i = first_empty_subgroup - 1; i > 0; --i) {
              count_type sg_size = subgroup_size[i];
              input_groups.emplace_back(input_type{gstart + local_left_border,
                                                   sg_size, gcontext, false, false});
              subgroup_size[i] = local_left_border;
              local_left_border += sg_size;
            }

            for (count_type i = 0; i < gsize; ++i) {
              subgroup_id[i] = subgroup_size[subgroup_id[i]]++;
            }
            for (count_type i = 0; i < gsize; ++i) {
              subgroup_size[subgroup_id[i]] = sa_interval[i];
            }
            for (count_type i = 0; i < gsize; ++i) {
              sa_interval[i] = subgroup_size[i];
            }
        }
    }
    else {
        if (!group.check_for_runs) {
          if (group.is_final) {
            // great! we can assign the rank!
            buffer_type const assign_rank = result_groups.size();
            #pragma omp parallel for
            for (count_type i = 0; i < gsize; ++i) {
              rank[F::remove_flag(sa_interval[i])] = assign_rank;
            }
            result_groups.emplace_back(output_type{gcontext, gsize});
          } else {
            // let's sort the group by the rank behind the context
            #pragma omp parallel for
            for (count_type i = 0; i < gsize; ++i) {
              to_sort[i].value = sa_interval[i];
            }
            #pragma omp parallel for
            for (count_type i = 0; i < gsize; ++i) {
              to_sort[i].key = rank[F::remove_flag(to_sort[i].value) + gcontext];
            }

            auto comp = [&](auto a, auto b) {
               return a.key > b.key || (a.key == b.key && a.value < b.value);
            };
            ips4o::parallel::sort(&(to_sort[0]), &(to_sort[gsize]), comp, threads);

            #pragma omp parallel for
            for (count_type i = 0; i < gsize; ++i) {
              sa_interval[i] = to_sort[i].value;
            }

            // calculate sg_count
            std::vector<count_type> sg_count_thread_vec(threads);
            count_type* sg_count_thread = sg_count_thread_vec.data();
            #pragma omp parallel for
            for (size_t i = 0; i < threads; ++i) {
                count_type interval_begin = i * (gsize / threads);
                count_type interval_end = i < threads-1 ? (count_type)((i + 1) * (gsize / threads)) : gsize;

                buffer_type sg_key = to_sort[interval_begin].key;
                sg_count_thread[i] = (((i == 0) || (to_sort[interval_begin-1].key != sg_key)) ? 1 : 0);
                for (count_type j = interval_begin+1; j < interval_end; ++j) {
                    if (to_sort[j].key != sg_key) {
                        ++sg_count_thread[i];
                        sg_key = to_sort[j].key;
                    }
                }
            }

            // prefix sum over sg_counts
            count_type sg_count = 0;
            for (size_t i = 0; i < threads; ++i) {
                count_type tmp = sg_count_thread[i];
                sg_count_thread[i] = sg_count;
                sg_count += tmp;
            }

            // calculate sg_borders
            #pragma omp parallel for
            for (size_t i = 0; i < threads; ++i) {
                count_type interval_begin = i * (gsize / threads);
                count_type interval_end = i < threads-1 ? (count_type)((i + 1) * (gsize / threads)) : gsize;

                buffer_type sg_key = to_sort[interval_begin].key;
                if ((i == 0) || (to_sort[interval_begin-1].key != sg_key)) {
                    to_sort[sg_count_thread[i]++].value = interval_begin; // using index value of to_sort for sg_borders
                }
                for (count_type j = interval_begin+1; j < interval_end; ++j) {
                    if (to_sort[j].key != sg_key) {
                        to_sort[sg_count_thread[i]++].value = j;
                        sg_key = to_sort[j].key;
                    }
                }
            }

            // emplace subgroups
            for (count_type i = 0; i < sg_count-1; ++i) {
                buffer_type sg_start = to_sort[i].value;
                count_type sg_size = to_sort[i+1].value-sg_start;
                buffer_type sg_key = to_sort[sg_start].key;
                buffer_type sg_context = gcontext + result_groups[sg_key].lyndon;
                input_groups.emplace_back(input_type{gstart + sg_start,
                                                     sg_size,
                                                     sg_context,
                                                     true,
                                                     false});
            }
            buffer_type sg_start = to_sort[sg_count-1].value;
            count_type sg_size = gsize-sg_start;
            buffer_type sg_key = to_sort[sg_start].key;
            buffer_type sg_context = gcontext + result_groups[sg_key].lyndon;
            input_groups.emplace_back(input_type{gstart + sg_start,
                                                 sg_size,
                                                 sg_context,
                                                 true,
                                                 false});
          }
        } else {
          // calculate subgroup_id and first_empty_subgroup
          uint8_t const uncertain_id = 2;
          buffer_type* length_end = subgroup_id + gsize;
          #pragma omp parallel for
          for (size_t i = 0; i < threads; ++i) {
              count_type interval_begin = i * (gsize / threads);
              count_type interval_end = i < threads-1 ? (count_type)((i + 1) * (gsize / threads)) : gsize;

              if (i != threads-1) {
                  subgroup_id[interval_end - 1] =
                          ((rank[F::remove_flag(sa_interval[interval_end - 1]) + gcontext])
                           ? ((buffer_type) 1) :
                           (((F::remove_flag(sa_interval[interval_end - 1]) + gcontext) != F::remove_flag(sa_interval[interval_end]))
                            ? ((buffer_type) 0) :
                            ((buffer_type) uncertain_id))); // we can't determine the subgroup yet
              }
              else {
                  subgroup_id[interval_end - 1] = (rank[F::remove_flag(sa_interval[interval_end - 1]) + gcontext]) ? (1)
                                                                                     : (0);
              }
              bool is_start = (subgroup_id[interval_end-1] == uncertain_id);
              length_end[i] = is_start;
              for (count_type j = interval_end-1; j > interval_begin; --j) {
                  subgroup_id[j - 1] =
                      ((rank[F::remove_flag(sa_interval[j - 1]) + gcontext])
                       ? ((buffer_type) 1) :
                       (((F::remove_flag(sa_interval[j - 1]) + gcontext) != F::remove_flag(sa_interval[j]))
                        ? ((buffer_type) 0) :
                          (is_start ? ((buffer_type) uncertain_id) :
                        ((subgroup_id[j]) ? (subgroup_id[j] + 1) : ((buffer_type) 0)))));
                  if ((is_start = (is_start && (subgroup_id[j - 1] == uncertain_id))))
                        ++length_end[i];
              }
          }
          for (size_t i = threads-2; i < threads-1; --i) {
              count_type interval_end = i < threads-1 ? (count_type)((i + 1) * (gsize / threads)) : gsize;
              buffer_type length = length_end[i];

              for (count_type j = interval_end; j > interval_end-length; --j) {
                  subgroup_id[j - 1] = ((subgroup_id[j]) ? (subgroup_id[j] + 1) : ((buffer_type) 0));
              }
          }
          size_t max_group = 0;
          #pragma omp parallel for reduction(max:max_group)
          for (count_type i = 0; i < gsize; ++i) {
              max_group = std::max(max_group, (size_t) subgroup_id[i]);
          }
          buffer_type first_empty_subgroup = max_group + 1;

          //calculate subgroup sizes
          buffer_type *const subgroup_size =
              (gsaca_likely(threads*first_empty_subgroup < max_group_size))
              ? (subgroup_id + gsize)
              : ((buffer_type *) malloc(threads*first_empty_subgroup * sizeof(buffer_type)));
          for (buffer_type i = 0; i < threads*first_empty_subgroup; ++i)
              subgroup_size[i] = 0;

          #pragma omp parallel for
          for (size_t i = 0; i < threads; ++i) {
              count_type interval_begin = i * (gsize / threads);
              count_type interval_end = i < threads-1 ? (count_type)((i + 1) * (gsize / threads)) : gsize;
              buffer_type* subgroup_size_thread = &(subgroup_size[i*first_empty_subgroup]);

              for (count_type j = interval_begin; j < interval_end; ++j) {
                  ++subgroup_size_thread[subgroup_id[j]];
              }
          }

          // calculate subgroup_borders
          count_type local_left_border = 0;
          for (size_t i = 0; i < threads; ++i) {
              buffer_type bucket_start = i*first_empty_subgroup;
              count_type tmp = subgroup_size[bucket_start];
              subgroup_size[bucket_start] = local_left_border;
              local_left_border += tmp;
          }
          for (buffer_type i = first_empty_subgroup - 1; i > 0; --i) {
              for (size_t j = 0; j < threads; ++j) {
                  buffer_type bucket_start = j*first_empty_subgroup+i;
                  buffer_type tmp = subgroup_size[bucket_start];
                  subgroup_size[bucket_start] = local_left_border;
                  local_left_border += tmp;
              }
          }

          // emplace subgroups
          buffer_type* subgroup_borders = subgroup_size;
          count_type sg_size = (first_empty_subgroup > 1) ? ((count_type) subgroup_borders[first_empty_subgroup-1]) : gsize;
          if (sg_size > 0) {
            input_groups.emplace_back(
                input_type{gstart, sg_size, gcontext, false, true});
          }
          for (count_type i = first_empty_subgroup - 1; i > 0; --i) {
            sg_size = (i > 1) ? ((count_type) (subgroup_borders[i-1]-subgroup_borders[i])) : (gsize-subgroup_borders[i]);
            input_groups.emplace_back(input_type{gstart + subgroup_borders[i],
                                                 sg_size, gcontext, false, false});
          }

          // distribute
          #pragma omp parallel for
          for (size_t i = 0; i < threads; ++i) {
              count_type interval_begin = i * (gsize / threads);
              count_type interval_end = i < threads-1 ? (count_type)((i + 1) * (gsize / threads)) : gsize;
              buffer_type* subgroup_size_thread = &(subgroup_size[i*first_empty_subgroup]);

              for (count_type j = interval_begin; j < interval_end; ++j) {
                  buffer_type &border = subgroup_size_thread[subgroup_id[j]];
                  subgroup_id[j] = border++;
              }
          }
          #pragma omp parallel for
          for (count_type i = 0; i < gsize; ++i) {
            subgroup_size[subgroup_id[i]] = sa_interval[i];
          }
          #pragma omp parallel for
          for (count_type i = 0; i < gsize; ++i) {
            sa_interval[i] = subgroup_size[i];
          }

          if (gsaca_unlikely(threads*first_empty_subgroup > max_group_size)) {
            free(subgroup_size);
          }
        }
    }
  }
  result_groups.emplace_back(output_type{n - 1, 1});
  result_groups.emplace_back(output_type{1, 1});
  sa[0] = n - 1;
  sa[1] = 0;
  std::reverse(result_groups.begin(), result_groups.end());
  result_groups.resize(result_groups.size() - 1);
  free(to_sort);
  return result_groups;
}

} // namespace gsaca_lyndon
