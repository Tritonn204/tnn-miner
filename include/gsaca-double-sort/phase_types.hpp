#pragma once

#include <deque>

namespace gsaca_lyndon {

template<typename buffer_type>
struct phase_1_group_type {
  static_assert(std::is_unsigned<buffer_type>::value);
  buffer_type start;
  buffer_type size;
  buffer_type context;
  bool check_for_runs;
  bool is_final;
};

template<typename index_type>
using phase_1_stack_type = std::deque<phase_1_group_type<index_type>>;


template<typename buffer_type>
struct phase_2_group_type {
  static_assert(std::is_unsigned<buffer_type>::value);
  buffer_type lyndon;
  buffer_type size;
};


}
