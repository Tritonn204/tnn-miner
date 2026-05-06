// hamilton.hip

#include "hip_platform.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>

// Kernel to check if a vertex is safe to add to the path
__global__ void isSafeKernel(int v,
                             const uint8_t* graph,
                             const uint16_t* path,
                             int pos,
                             int* result,
                             int V)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if vertex v is already in the path
    if (tid < pos) {
        if (path[tid] == v) {
            atomicAnd(result, 0);
        }
    }

    // Check edge from previous vertex to v
    if (tid == 0) {
        if (pos == 0 || graph[path[pos - 1] * V + v] == 0) {
            atomicAnd(result, 0);
        }
    }
}

// Host function to find Hamiltonian cycle
// Note: input graph is still vector<vector<bool>> for convenience.
std::vector<uint16_t> findHamiltonianCycle(const std::vector<std::vector<bool>>& graph)
{
    int V = static_cast<int>(graph.size());
    if (V == 0) {
        return {};
    }

    // Path initialized to "invalid" value, but we really only care about [0..pos)
    std::vector<uint16_t> path(V, UINT16_MAX);
    path[0] = 0;

    // Allocate device memory
    uint8_t* d_graph = nullptr;
    uint16_t* d_path = nullptr;
    int* d_result = nullptr;

    if (hipMalloc(&d_graph, V * V * sizeof(uint8_t)) != hipSuccess ||
        hipMalloc(&d_path, V * sizeof(uint16_t)) != hipSuccess ||
        hipMalloc(&d_result, sizeof(int)) != hipSuccess) {
        std::cerr << "hipMalloc failed\n";
        if (d_graph) hipFree(d_graph);
        if (d_path) hipFree(d_path);
        if (d_result) hipFree(d_result);
        return {};
    }

    // Flatten graph into a contiguous byte array (0 / 1)
    std::vector<uint8_t> flattened_graph(V * V);
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            flattened_graph[i * V + j] = graph[i][j] ? 1u : 0u;
        }
    }

    // Copy graph to device
    if (hipMemcpy(d_graph,
                  flattened_graph.data(),
                  V * V * sizeof(uint8_t),
                  hipMemcpyHostToDevice) != hipSuccess) {
        std::cerr << "hipMemcpy (graph) failed\n";
        hipFree(d_graph);
        hipFree(d_path);
        hipFree(d_result);
        return {};
    }

    // Main loop to build Hamiltonian cycle
    for (int pos = 1; pos < V; ++pos) {
        bool found = false;

        for (int v = 1; v < V; ++v) {
            // Assume vertex v is usable until kernel disproves it
            int result = 1;

            if (hipMemcpy(d_result, &result, sizeof(int),
                          hipMemcpyHostToDevice) != hipSuccess ||
                hipMemcpy(d_path, path.data(), V * sizeof(uint16_t),
                          hipMemcpyHostToDevice) != hipSuccess) {
                std::cerr << "hipMemcpy (path/result) failed\n";
                hipFree(d_graph);
                hipFree(d_path);
                hipFree(d_result);
                return {};
            }

            int blockSize = 256;
            int gridSize  = (pos + blockSize - 1) / blockSize;

            TNN_HIP_LAUNCH_KERNEL(isSafeKernel,
                               dim3(gridSize),
                               dim3(blockSize),
                               0,
                               0,
                               v,
                               d_graph,
                               d_path,
                               pos,
                               d_result,
                               V);

            if (hipDeviceSynchronize() != hipSuccess) {
                std::cerr << "hipDeviceSynchronize failed\n";
                hipFree(d_graph);
                hipFree(d_path);
                hipFree(d_result);
                return {};
            }

            if (hipMemcpy(&result, d_result, sizeof(int),
                          hipMemcpyDeviceToHost) != hipSuccess) {
                std::cerr << "hipMemcpy (result back) failed\n";
                hipFree(d_graph);
                hipFree(d_path);
                hipFree(d_result);
                return {};
            }

            if (result) {
                path[pos] = static_cast<uint16_t>(v);
                found = true;
                break;
            }
        }

        if (!found) {
            // Clean up and fail
            hipFree(d_graph);
            hipFree(d_path);
            hipFree(d_result);
            return {};
        }
    }

    // We can free GPU memory before the final CPU-side check
    hipFree(d_graph);
    hipFree(d_path);
    hipFree(d_result);

    // Check if there's an edge from the last vertex back to the first
    if (graph[path[V - 1]][path[0]]) {
        return path;
    }

    return {};
}

int main()
{
    // Example usage
    std::vector<std::vector<bool>> graph = {
        {0, 1, 0, 1, 0},
        {1, 0, 1, 1, 1},
        {0, 1, 0, 0, 1},
        {1, 1, 0, 0, 1},
        {0, 1, 1, 1, 0}
    };

    std::vector<uint16_t> result = findHamiltonianCycle(graph);

    if (result.empty()) {
        std::cout << "No Hamiltonian cycle found.\n";
    } else {
        std::cout << "Hamiltonian cycle: ";
        for (uint16_t v : result) {
            std::cout << v << " ";
        }
        // close the cycle
        std::cout << result[0] << "\n";
    }

    return 0;
}
