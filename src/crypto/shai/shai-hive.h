#include <random>
#include <array>
#include <util/hamilton/hamilton.hpp>

namespace ShaiHive {
  constexpr int SHAI_DATA_SIZE = 80; // data is 76, nonce is 4
  constexpr int GRAPH_SIZE = 2008; // V2 max graph size

  inline uint32_t flipped32(uint32_t in)
  {
    return __builtin_bswap32(in);
  }

  inline bool checkNonce(uint32_t *Xr, uint32_t *Yr)
  {
    uint32_t X[8] = {
        flipped32(Xr[7]), flipped32(Xr[6]),
        flipped32(Xr[5]), flipped32(Xr[4]),
        flipped32(Xr[3]), flipped32(Xr[2]),
        flipped32(Xr[1]), flipped32(Xr[0])};
    uint32_t Y[8] = {
        flipped32(Yr[7]), flipped32(Yr[6]),
        flipped32(Yr[5]), flipped32(Yr[4]),
        flipped32(Yr[3]), flipped32(Yr[2]),
        flipped32(Yr[1]), flipped32(Yr[0])};

    return (X[7] != Y[7] ? X[7] < Y[7] : X[6] != Y[6] ? X[6] < Y[6]
                                     : X[5] != Y[5]   ? X[5] < Y[5]
                                     : X[4] != Y[4]   ? X[4] < Y[4]
                                     : X[3] != Y[3]   ? X[3] < Y[3]
                                     : X[2] != Y[2]   ? X[2] < Y[2]
                                     : X[1] != Y[1]   ? X[1] < Y[1]
                                                      : X[0] < Y[0]);
  }

  typedef struct alignas(64) ShaiCtx{
    uint8_t data[SHAI_DATA_SIZE + GRAPH_SIZE*2];
    alignas(64) uint8_t sha[32];
    alignas(64) uint16_t freev[GRAPH_SIZE];
    // std::vector<std::vector<bool>> graph;
    HamiltonGraph::ShaiGraph packedGraph;
    // uint8_t gPad[(GRAPH_SIZE*GRAPH_SIZE + 64)/8];
    // std::vector<uint16_t> path;
    uint16_t path[GRAPH_SIZE];
    uint8_t padding[4096];
  } ShaiCtx;

  void tuneTimeLimit();
  bool hash(ShaiCtx &ctx, uint8_t *data);
  void test();
}