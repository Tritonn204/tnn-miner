#pragma once

#include <vector>
#include <queue>
#include <atomic>
#include <mutex>
#include <bitset>
#include <array>
#include <random>
#include <algorithm>
#include <optional>
#include <iostream>
#include <string>
#include <sstream>
#include <unordered_set>

namespace HamiltonGraph
{
  extern std::chrono::time_point<std::chrono::_V2::steady_clock, std::chrono::duration<long long int, std::ratio<1, 1000000000>>> start_time;

  typedef struct alignas(32) Node
  {
    std::vector<uint16_t> solution;
    int level;

    Node() : level(-1) {} // Default constructor

    explicit Node(int start_vertex) : level(0)
    {
      solution.push_back(start_vertex);
    }
  } Node;

  class ThreadSafeQueue
  {
  private:
    std::queue<Node> queue;
    mutable std::mutex mutex;

  public:
    void enqueue(Node node)
    {
      std::scoped_lock<std::mutex> lock(mutex);
      queue.push(std::move(node));
    }

    bool dequeue(Node &node)
    {
      std::scoped_lock<std::mutex> lock(mutex);
      if (queue.empty())
        return false;
      node = std::move(queue.front());
      queue.pop();
      return true;
    }

    bool empty() const
    {
      std::scoped_lock<std::mutex> lock(mutex);
      return queue.empty();
    }
  };

  static constexpr size_t PACKED_GRAPH_SIZE = ((2008 * (2008 - 1) / 2) + 31) / 32;

  struct alignas(64) ShaiGraph
  {
    ShaiGraph() : _size(2008)
    {
    }

    ShaiGraph(int size, bool v) : _size(size)
    {
    }

    alignas(64) uint32_t d[PACKED_GRAPH_SIZE];
    int _size;

    inline int coord_to_idx(int i, int j) const
    {
      if (i <= j)
        return i * _size - ((i * (i + 1)) >> 1) + (j - (i + 1));

      return j * _size - ((j * (j + 1)) >> 1) + (i - (j + 1));
    }

    inline bool get(int i, int j) const
    {
      int idx = coord_to_idx(i, j);
      int bit_pos = idx & 0x1f;

      // if (idx >> 5 >= PACKED_GRAPH_SIZE) {
      //   printf("idx = %d, upper bound = %d\n", idx >> 5, PACKED_GRAPH_SIZE);
      //   fflush(stdout);
      // }

      uint32_t val = (d[idx >> 5] & (1 << (31 - bit_pos)));

      return val;
    }

    inline int size() const { return _size; }
  };

  bool solveGraph(std::vector<std::vector<bool>> &graph,
                            std::vector<uint16_t> &path,
                            size_t pos,
                            int timeLimitMS = 500);

  bool solveGraph(ShaiGraph &graph,
                          std::vector<uint16_t> &path,
                          size_t pos,
                          int timeLimitMS = 500);

  bool solveGraph(HamiltonGraph::ShaiGraph &graph,
                  std::vector<uint16_t> &path,
                  size_t pos,
                  uint16_t *freev,
                  int &nchecks
                  );

  bool solveGraph(ShaiGraph &graph,
                  uint16_t *path,
                  size_t pos,
                  uint16_t *freev,
                  int timeLimitMS);

  bool solveGraph_opt(HamiltonGraph::ShaiGraph &graph,
                  uint16_t *path,
                  size_t pos,
                  uint16_t *freev,
                  int &nchecks);

  std::vector<uint16_t> findHamiltonianCycle_V2(std::vector<std::vector<bool>> graph);
}