#include "hamilton.hpp"
#include <thread>
#include <iostream>
#include <algorithm>
#if defined(__x86_64__)
  #include <immintrin.h>
#elif defined(__aarch64__)
  #include <arm_neon.h>
#endif
#include <stack>
#include <tuple>
#include <vector>

namespace HamiltonGraph
{
  std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<long long int, std::ratio<1, 1000000000>>> start_time = std::chrono::steady_clock::now();

#if defined(__x86_64__)
  __attribute__((target("default"))) bool isSafe(int v,
                                                 const std::vector<std::vector<bool>> &graph,
                                                 std::vector<uint16_t> &path,
                                                 int pos)
  {
    if (pos == 0 || !graph[path[pos - 1]][v])
    {
      return false;
    }

    for (int i = 0; i < pos; i++)
    {
      if (path[i] == v)
      {
        return false;
      }
    }

    return true;
  }

  __attribute__((target("default"))) bool isSafe_packed(int v,
                                                 const ShaiGraph &graph,
                                                 std::vector<uint16_t> &path,
                                                 int pos)
  {
    if (pos == 0 || !graph.get(path[pos - 1],v))
    {
      return false;
    }

    for (int i = 0; i < pos; i++)
    {
      if (path[i] == v)
      {
        return false;
      }
    }

    return true;
  }

  __attribute__((target("sse2"))) bool isSafe(int v,
                                              const std::vector<std::vector<bool>> &graph,
                                              std::vector<uint16_t> &path,
                                              int pos)
  {
    if (pos == 0 || !graph[path[pos - 1]][v])
    {
      return false;
    }

    const int simd_width = 8; // SSE2: 128 bits = 16 bytes = 8 uint16_t elements
    __m128i v_vec = _mm_set1_epi16(v);

    int i = 0;
    for (; i <= pos - simd_width; i += simd_width)
    {
      __m128i path_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&path[i]));
      __m128i cmp = _mm_cmpeq_epi16(path_vec, v_vec);

      if (_mm_movemask_epi8(cmp) != 0)
      {
        return false;
      }
    }

    // Handle remaining elements
    for (; i < pos; ++i)
    {
      if (path[i] == v)
      {
        return false;
      }
    }

    return true;
  }

  __attribute__((target("sse2"))) bool isSafe_packed(int v,
                                              const ShaiGraph &graph,
                                              std::vector<uint16_t> &path,
                                              int pos)
  {
    if (pos == 0 || !graph.get(path[pos - 1],v))
    {
      return false;
    }

    const int simd_width = 8; // SSE2: 128 bits = 16 bytes = 8 uint16_t elements
    __m128i v_vec = _mm_set1_epi16(v);

    int i = 0;
    for (; i <= pos - simd_width; i += simd_width)
    {
      __m128i path_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&path[i]));
      __m128i cmp = _mm_cmpeq_epi16(path_vec, v_vec);

      if (_mm_movemask_epi8(cmp) != 0)
      {
        return false;
      }
    }

    // Handle remaining elements
    for (; i < pos; ++i)
    {
      if (path[i] == v)
      {
        return false;
      }
    }

    return true;
  }

  // AVX2-optimized version
  __attribute__((target("avx2"))) bool isSafe(int v,
                                              const std::vector<std::vector<bool>> &graph,
                                              std::vector<uint16_t> &path,
                                              int pos)
  {
    if (pos == 0 || !graph[path[pos - 1]][v])
    {
      return false;
    }

    const int simd_width = 16; // AVX2: 256 bits = 16 bytes = 8 uint16_t elements
    __m256i v_vec = _mm256_set1_epi16(v);

    int i = 0;
    for (; i <= pos - simd_width; i += simd_width)
    {
      __m256i path_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&path[i]));
      __m256i cmp = _mm256_cmpeq_epi16(path_vec, v_vec);

      if (_mm256_movemask_epi8(cmp) != 0)
      {
        return false;
      }
    }

    for (; i < pos; ++i)
    {
      if (path[i] == v)
      {
        return false;
      }
    }

    return true;
  }

  // AVX2-optimized version
  __attribute__((target("avx2"))) bool isSafe_packed(int v,
                                              const ShaiGraph &graph,
                                              std::vector<uint16_t> &path,
                                              int pos)
  {
    if (pos == 0 || !graph.get(path[pos - 1],v))
    {
      return false;
    }

    const int simd_width = 16; // AVX2: 256 bits = 16 bytes = 8 uint16_t elements
    __m256i v_vec = _mm256_set1_epi16(v);

    int i = 0;
    for (; i <= pos - simd_width; i += simd_width)
    {
      __m256i path_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&path[i]));
      __m256i cmp = _mm256_cmpeq_epi16(path_vec, v_vec);

      if (_mm256_movemask_epi8(cmp) != 0)
      {
        return false;
      }
    }

    for (; i < pos; ++i)
    {
      if (path[i] == v)
      {
        return false;
      }
    }

    return true;
  }

  // AVX-512-optimized version
  __attribute__((target("avx512f,avx512bw"))) bool isSafe(int v,
                                                          const std::vector<std::vector<bool>> &graph,
                                                          std::vector<uint16_t> &path,
                                                          int pos)
  {
    if (pos == 0 || !graph[path[pos - 1]][v])
    {
      return false;
    }

    const int simd_width = 32; // AVX-512: 512 bits = 32 bytes = 16 uint16_t elements
    __m512i v_vec = _mm512_set1_epi16(v);

    int i = 0;
    for (; i <= pos - simd_width; i += simd_width)
    {
      __m512i path_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&path[i]));
      __mmask32 cmp_mask = _mm512_cmpeq_epi16_mask(path_vec, v_vec);

      if (cmp_mask != 0)
      {
        return false;
      }
    }

    for (; i < pos; ++i)
    {
      if (path[i] == v)
      {
        return false;
      }
    }

    return true;
  }

  // // AVX-512-optimized version
  // __attribute__((target("avx512f,avx512bw"))) bool isSafe_packed(int v,
  //                                                         ShaiGraph &graph,
  //                                                         std::vector<uint16_t> &path,
  //                                                         int pos)
  // {
  //   if (pos == 0 || !graph.get(path[pos - 1],v))
  //   {
  //     return false;
  //   }

  //   const int simd_width = 32; // AVX-512: 512 bits = 32 bytes = 16 uint16_t elements
  //   __m512i v_vec = _mm512_set1_epi16(v);

  //   int i = 0;
  //   for (; i <= pos - simd_width; i += simd_width)
  //   {
  //     __m512i path_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&path[i]));
  //     __mmask32 cmp_mask = _mm512_cmpeq_epi16_mask(path_vec, v_vec);

  //     if (cmp_mask != 0)
  //     {
  //       return false;
  //     }
  //   }

  //   for (; i < pos; ++i)
  //   {
  //     if (path[i] == v)
  //     {
  //       return false;
  //     }
  //   }

  //   return true;
  // }

#elif defined(__aarch64__)
  bool isSafe(int v,
              const std::vector<std::vector<bool>> &graph,
              std::vector<uint16_t> &path,
              int pos)
  {
    if (pos == 0 || !graph[path[pos - 1]][v])
    {
      return false;
    }

    const int simd_width = 8; // NEON: 128 bits = 16 bytes = 8 uint16_t elements
    uint16x8_t v_vec = vdupq_n_u16(v);

    int i = 0;
    for (; i <= pos - simd_width; i += simd_width)
    {
      uint16x8_t path_vec = vld1q_u16(&path[i]);
      uint16x8_t cmp = vceqq_u16(path_vec, v_vec);

      if (vmaxvq_u16(cmp) != 0) // vmaxvq_u16 reduces to check if any match is found
      {
        return false;
      }
    }

    // Handle remaining elements
    for (; i < pos; ++i)
    {
      if (path[i] == v)
      {
        return false;
      }
    }

    return true;
  }

  bool isSafe_packed(int v,
              ShaiGraph &graph,
              std::vector<uint16_t> &path,
              int pos)
  {
    if (pos == 0 || !graph.get(path[pos - 1],v))
    {
      return false;
    }

    const int simd_width = 8; // NEON: 128 bits = 16 bytes = 8 uint16_t elements
    uint16x8_t v_vec = vdupq_n_u16(v);

    int i = 0;
    for (; i <= pos - simd_width; i += simd_width)
    {
      uint16x8_t path_vec = vld1q_u16(&path[i]);
      uint16x8_t cmp = vceqq_u16(path_vec, v_vec);

      if (vmaxvq_u16(cmp) != 0) // vmaxvq_u16 reduces to check if any match is found
      {
        return false;
      }
    }

    // Handle remaining elements
    for (; i < pos; ++i)
    {
      if (path[i] == v)
      {
        return false;
      }
    }

    return true;
  }
#endif
  bool solveGraph(std::vector<std::vector<bool>> &graph,
                            std::vector<uint16_t> &path,
                            size_t pos,
                            int timeLimitMS)
  {
    std::vector<bool> visited(graph.size(), false);
    visited[0] = true;

    std::stack<std::pair<size_t, size_t>> position_vertex_stack;
    size_t vertex = 1;
    position_vertex_stack.push({pos, vertex});

    auto ST = std::chrono::steady_clock::now();

    while (true)
    {
      auto current_time = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::milliseconds>(
              current_time - ST)
              .count() > timeLimitMS)
      {
        return false;
      }

      if (pos == graph.size())
      {
        if (graph[path[pos - 1]][path[0]])
        {
          return true;
        }

        if (!position_vertex_stack.empty())
        {
          auto [prev_pos, prev_vertex] = position_vertex_stack.top();
          position_vertex_stack.pop();

          visited[path[prev_pos]] = false;
          path[prev_pos] = -1;
          pos = prev_pos;
          vertex = prev_vertex + 1;
          continue;
        }
        return false;
      }

      while (vertex < graph.size())
      {
        if (!visited[vertex] && isSafe(vertex, graph, path, pos))
        {
          path[pos] = vertex;
          visited[vertex] = true;
          position_vertex_stack.push({pos, vertex});
          pos++;
          vertex = 1;
          break;
        }
        vertex++;
      }

      if (vertex >= graph.size())
      {
        if (!position_vertex_stack.empty())
        {
          auto [prev_pos, prev_vertex] = position_vertex_stack.top();
          position_vertex_stack.pop();
          visited[path[prev_pos]] = false;
          path[prev_pos] = -1;
          pos = prev_pos;
          vertex = prev_vertex + 1;
        }
        else
        {
          return false;
        }
      }
    }
  }

  bool solveGraph_opt(HamiltonGraph::ShaiGraph &graph,
                  uint16_t *path,
                  size_t pos,
                  uint16_t *freev,
                  int &nchecks)
  {
    nchecks++;
    if (nchecks > 7000)
    {
      return false;
    }
    if (pos == graph.size())
    {
      if (graph.get(path[pos - 1], path[0]))
      {
        return true;
      }
      else
      {
        return false;
      }
    }

    uint16_t prev_pos = 0;

    while (true)
    {
      uint16_t v = freev[prev_pos];
      if (v == graph.size())
        break;

      if (graph.get(path[pos - 1], v))
      {
        path[pos] = v;
        freev[prev_pos] = freev[v];

        if (solveGraph_opt(graph, path, pos + 1, freev, nchecks))
        {
          return true;
        }
        if (nchecks > 7000)
        {
          return false;
        }

        freev[prev_pos] = v;
        path[pos] = -1;
      }

      prev_pos = v;
    }

    return false;
  }

  bool solveGraph(ShaiGraph &graph,
                  uint16_t *path,
                  size_t pos,
                  uint16_t *freev,
                  int timeLimitMS)
  {
    std::stack<std::tuple<size_t, uint16_t, uint16_t>> state_stack; // pos, prev_pos, v
    uint16_t prev_pos = 0;

    int nchecks = 0;
    auto ST = std::chrono::steady_clock::now();

    while (true)
    {
      nchecks++;  // Increment check counter
      if (nchecks > 7000) {
          return false;  // Exceeded check limit
      }

      if (pos == graph.size())
      {
        if (graph.get(path[pos - 1], path[0]))
        {
          return true;
        }

        if (!state_stack.empty())
        {
          auto [old_pos, old_prev_pos, old_v] = state_stack.top();
          state_stack.pop();

          // Restore the linked list state
          freev[old_prev_pos] = old_v;
          path[old_pos] = -1;

          pos = old_pos;
          prev_pos = old_v; // Same change here - move to next in list
          continue;
        }
        return false;
      }

      uint16_t v = freev[prev_pos];
      while (v != graph.size())
      {
        if (graph.get(path[pos - 1], v))
        {
          path[pos] = v;
          // Save state before modifying freev
          state_stack.push({pos, prev_pos, v});

          // Remove v from available vertices
          freev[prev_pos] = freev[v];

          pos++;
          prev_pos = 0; // Reset for next level
          break;
        }
        prev_pos = v;
        v = freev[prev_pos];
      }

      if (v == graph.size())
      { // No valid vertex found
        if (!state_stack.empty())
        {
          auto [old_pos, old_prev_pos, old_v] = state_stack.top();
          state_stack.pop();

          // Restore the linked list state
          freev[old_prev_pos] = old_v;
          path[old_pos] = -1;

          pos = old_pos;
          prev_pos = old_v; // Start from the vertex we just tried
                            // This effectively moves us to the next available vertex in the list
        }
        else
        {
          return false;
        }
      }
    }
  }

  bool solveGraph(ShaiGraph &graph,
                            std::vector<uint16_t> &path,
                            size_t pos,
                            int timeLimitMS)
  {
    std::vector<bool> visited(graph.size(), false);
    visited[0] = true;

    std::stack<std::pair<size_t, size_t>> position_vertex_stack;
    size_t vertex = 1;
    position_vertex_stack.push({pos, vertex});

    auto ST = std::chrono::steady_clock::now();

    while (true)
    {
      auto current_time = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::milliseconds>(
              current_time - ST)
              .count() > timeLimitMS)
      {
        return false;
      }

      if (pos == graph.size())
      {
        if (graph.get(path[pos - 1], path[0]))
        {
          return true;
        }

        if (!position_vertex_stack.empty())
        {
          auto [prev_pos, prev_vertex] = position_vertex_stack.top();
          position_vertex_stack.pop();

          visited[path[prev_pos]] = false;
          path[prev_pos] = -1;
          pos = prev_pos;
          vertex = prev_vertex + 1;
          continue;
        }
        return false;
      }

      while (vertex < graph.size())
      {
        if (!visited[vertex] && isSafe_packed(vertex, graph, path, pos))
        {
          path[pos] = vertex;
          visited[vertex] = true;
          position_vertex_stack.push({pos, vertex});
          pos++;
          vertex = 1;
          break;
        }
        vertex++;
      }

      if (vertex >= graph.size())
      {
        if (!position_vertex_stack.empty())
        {
          auto [prev_pos, prev_vertex] = position_vertex_stack.top();
          position_vertex_stack.pop();
          visited[path[prev_pos]] = false;
          path[prev_pos] = -1;
          pos = prev_pos;
          vertex = prev_vertex + 1;
        }
        else
        {
          return false;
        }
      }
    }
  }  

  std::vector<uint16_t> findHamiltonianCycle_V2(std::vector<std::vector<bool>> graph)
  {
    std::vector<uint16_t> path(graph.size(), -1);
    path[0] = 0;

    if (!solveGraph(graph, path, 1))
    {
      return {};
    }
    return path;
  }
}