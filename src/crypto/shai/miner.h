// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2015 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_MINER_H
#define BITCOIN_MINER_H

#include "primitives/block.h"
#include <validation.h>
#include <stdint.h>
#include <net.h>
#include <random>

using Clock = std::chrono::steady_clock;

class CBlockIndex;
class CChainParams;
class CReserveKey;
class CScript;
class CWallet;
namespace Consensus { struct Params; };

class HCGraphUtil {
    std::chrono::time_point<Clock> startTime;

    template<typename T>
    T hexToType(const std::string& hexString)
    {
        static_assert(std::is_integral<T>::value, "Integral type required.");
        T number;
        std::stringstream ss;
        ss << std::hex << hexString;
        ss >> number;
        return number;
    }

    uint64_t extractSeedFromHash(const uint256& hash)
    {
        return hash.GetUint64(0);
    }

    public: 


    bool static verifyHamiltonianCycle(const std::vector<std::vector<bool>>& graph,
                                       const std::array<uint16_t, GRAPH_SIZE>& path)
    {
        size_t path_size = 0;
        auto it = std::find(path.begin(), path.end(), USHRT_MAX);
        if (it != path.end()) {
            path_size = std::distance(path.begin(), it);
        }

        size_t n = graph.size();

        // Check if path contains all vertices exactly once
        if (path_size != n) {
            return false;
        }
        std::unordered_set<uint16_t> verticesInPath(path.begin(), path.begin() + path_size);
        if (verticesInPath.size() != n) {
            return false;
        }

        // Check if the path forms a cycle
        for (size_t i = 1; i < n; ++i) {
            if (!graph[path[i - 1]][path[i]]) {
                return false;
            }
        }

        // Check if there's an edge from the last to the first vertex to form a cycle
        if (!graph[path[n - 1]][path[0]]) {
            return false;
        }
        
        return true;
    }

    uint16_t getGridSize(const std::string& hash)
    {
        int minGridSize = 512;
        int maxGridSize = GRAPH_SIZE;
        std::string gridSizeSegment = hash.substr(0, 8);
        unsigned long long gridSize = hexToType<unsigned long long>(gridSizeSegment);

        // Normalize gridSize to within the range
        int normalizedGridSize = minGridSize + (gridSize % (maxGridSize - minGridSize));

        // Adjust to hit maxGridSize more frequently
        if ((gridSize % 8) == 0)
        {
            normalizedGridSize = maxGridSize;
        }
        return normalizedGridSize;
    }

    uint16_t getGridSize_V2(const std::string& hash)
    {
        int min_grid_size = 2000;
        int max_grid_size = GRAPH_SIZE;
        std::string grid_size_segment = hash.substr(0, 8);
        unsigned long long grid_size = hexToType<unsigned long long>(grid_size_segment);
        auto grid_size_final = min_grid_size + (grid_size % (max_grid_size - min_grid_size));
        if(grid_size_final > GRAPH_SIZE) {
            grid_size_final = GRAPH_SIZE;
        }
        return grid_size_final;
    }

    std::vector<std::vector<bool>> generateGraph(const uint256& hash,
                                                 uint16_t gridSize)
    {
        std::vector<std::vector<bool>> graph(gridSize, std::vector<bool>(gridSize, false));
        int hashLength = hash.size();
        std::string ref_hash_index = hash.ToString();
        for (size_t i = 0; i < gridSize; ++i) {
            for (int j = i + 1; j < gridSize; ++j) {
                int hashIndex = (i * gridSize + j) * 2 % hashLength;
                uint8_t ch1 = ref_hash_index[hashIndex % hashLength];
                uint8_t ch2 = ref_hash_index[(hashIndex + 1) % hashLength];

                unsigned int edgeValue = ((isdigit(ch1) ? ch1 - '0' : ch1 - 'a' + 10) << 4) +
                                        (isdigit(ch2) ? ch2 - '0' : ch2 - 'a' + 10);
                if (edgeValue < 128) {
                    graph[i][j] = graph[j][i] = true;
                }
            }
        }
        return graph;
    }

    std::vector<std::vector<bool>> generateGraph_V2(const uint256& hash,
                                                 uint16_t gridSize)
    {
        std::vector<std::vector<bool>> graph(gridSize, std::vector<bool>(gridSize, false));
        size_t numEdges = (gridSize * (gridSize - 1)) / 2;
        size_t bitsNeeded = numEdges; // One bit per edge

        // Extract seed from hash
        uint64_t seed = extractSeedFromHash(hash);

        // Initialize PRNG with seed
        std::mt19937_64 prng;
        prng.seed(seed);

        // Generate bitsNeeded bits
        std::vector<bool> bitStream;
        bitStream.reserve(bitsNeeded);

        for (size_t i = 0; i < bitsNeeded; ++i) {
            uint32_t randomBits = prng();
            // Extract bits from randomBits
            for (int j = 31; j >= 0 && bitStream.size() < bitsNeeded; --j) {
                bool bit = (randomBits >> j) & 1;
                bitStream.push_back(bit);
            }
        }

        // Fill the adjacency matrix
        size_t bitIndex = 0;
        for (size_t i = 0; i < gridSize; ++i) {
            for (size_t j = i + 1; j < gridSize; ++j) {
                bool edgeExists = bitStream[bitIndex++];
                graph[i][j] = graph[j][i] = edgeExists;
            }
        }
        return graph;
    }
    static const size_t PACKED_GRAPH_SIZE = ((GRAPH_SIZE*GRAPH_SIZE / 2) + 31) / 32;

    struct GraphData {
        GraphData(int size, bool v) : _size(size) { 
         }

        uint32_t d[PACKED_GRAPH_SIZE];
        int _size;

        inline int coord_to_idx(int i, int j) const {
            if (i <= j)
                return i * _size - ((i*(i + 1)) >> 1) + (j-(i+1)); 

            return j * _size - ((j*(j + 1)) >> 1) + (i - (j+1));            
        }

        inline bool get(int i, int j) const {
            int idx = coord_to_idx(i,j);
            int bit_pos = idx & 0x1f;

            uint32_t val = (d[idx >> 5] & (1 << (31-bit_pos)));

            return val; 
        }

        
        inline int size() const { return _size; }
    };


    GraphData MygenerateGraph_V2(const uint256& hash,
                                                 uint16_t gridSize)
    {
        GraphData graph(gridSize, false);

        // Extract seed from hash
        uint64_t seed = extractSeedFromHash(hash);

        // Initialize PRNG with seed
        std::mt19937_64 prng;
        prng.seed(seed);

        // Fill the adjacency matrix
        for(size_t dw=0; dw<PACKED_GRAPH_SIZE; dw++) {

            graph.d[dw] = prng();
        }
        return graph;
    }

    int nchecks;
    const int MAX_CHECKS = 7000;
    bool hamiltonianCycleUtil(GraphData& graph,
                              std::vector<uint16_t>& path,
                              size_t pos, 
                              std::vector<uint16_t>& freev
                              )
    {
        nchecks++;
        if (nchecks > MAX_CHECKS) {
            return false;
        }

        if (pos == graph.size()) {
            if (graph.get(path[pos - 1], path[0])) {
                return true;
            } else {
                return false;
            }
        }

        uint16_t prev_pos = 0;

        while(true) {
            uint16_t v = freev[prev_pos];
            if (v == graph.size())
                break;        
            
            if (graph.get(path[pos - 1], v)) {
                path[pos] = v;
                freev[prev_pos] = freev[v];
                
                if (hamiltonianCycleUtil(graph, path, pos + 1, freev)) {
                    return true;
                }
                if (nchecks > MAX_CHECKS) {
                    return false;
                }                
                

                freev[prev_pos] = v;
                path[pos] = -1;                
            }

            prev_pos = v;
        }

        return false;
    }

    std::vector<uint16_t> findHamiltonianCycle(uint256 graph_hash)
    {
        return {};
    }

    std::vector<uint16_t> findHamiltonianCycle_V2(uint256 graph_hash)
    {
        nchecks=0;
        GraphData graph = MygenerateGraph_V2(graph_hash, getGridSize_V2(graph_hash.ToString()));
        std::vector<uint16_t> path(graph.size(), -1);
        
        std::vector<uint16_t> freev(graph.size(), 0);
        for(int i=0; i<freev.size(); i++)
            freev[i]=i+1;

        path[0] = 0;
        startTime = Clock::now();

        if (!hamiltonianCycleUtil(graph, path, 1, freev)) {
            return {};
        }
        return path;
    }

};

/** Run the miner threads */
void GenerateShaicoins(std::optional<CScript> minerAddress,
                       const CChainParams& chainparams,
                       ChainstateManager& chainman,
                       const CConnman& conman,
                       const CTxMemPool& mempool);

#endif // BITCOIN_MINER_H