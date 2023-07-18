#pragma once

#include <cstring>
#include "phase_2.hpp"
#include "../phase_types.hpp"
#include "../radix32.hpp"


namespace gsaca_lyndon {

template<typename sorter, typename F = flag_type<false>,
    typename index_type, typename buffer_type>
inline auto phase_1_by_sorting(index_type *const sa, buffer_type *const isa,
                               phase_1_stack_type<buffer_type> &input_groups) {
  using count_type = get_count_type<index_type, buffer_type>;
  using output_type = phase_2_group_type<buffer_type>;
  using input_type = phase_1_group_type<buffer_type>;
  using sorting_type = radix_key_val_pair<buffer_type>;

  size_t max_group_size = 0;
  for (auto const &group : input_groups) {
    max_group_size = std::max(max_group_size, (size_t) group.size);
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
          to_sort[i].key = rank[F::remove_flag(to_sort[i].value) + gcontext];
        }

        size_t max_rank = result_groups.size() - 1;
        // decreasing sort, stable sort
        sorter::template sort<false, true>(to_sort, to_sort + gsize, gsize,
                                           max_rank);

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
