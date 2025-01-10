#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

// Kernel to check if a vertex is safe to add to the path
__global__ void isSafeKernel(int v, const bool* graph, const uint16_t* path, int pos, int* result, int V) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < pos) {
        if (path[tid] == v) {
            atomicAnd(result, 0);
        }
    }
    
    if (tid == 0 && (pos == 0 || graph[path[pos-1] * V + v] == 0)) {
        atomicAnd(result, 0);
    }
}

// Host function to find Hamiltonian cycle
std::vector<uint16_t> findHamiltonianCycle(const std::vector<std::vector<bool>>& graph) {
    int V = graph.size();
    std::vector<uint16_t> path(V, -1);
    path[0] = 0;

    // Allocate device memory
    bool* d_graph;
    uint16_t* d_path;
    int* d_result;

    hipMalloc(&d_graph, V * V * sizeof(bool));
    hipMalloc(&d_path, V * sizeof(uint16_t));
    hipMalloc(&d_result, sizeof(int));

    // Copy graph to device
    std::vector<bool> flattened_graph;
    for (const auto& row : graph) {
        flattened_graph.insert(flattened_graph.end(), row.begin(), row.end());
    }
    hipMemcpy(d_graph, flattened_graph.data(), V * V * sizeof(bool), hipMemcpyHostToDevice);

    // Main loop to find Hamiltonian cycle
    for (int pos = 1; pos < V; ++pos) {
        bool found = false;
        for (int v = 1; v < V; ++v) {
            // Check if vertex v can be added to the path
            int result = 1;
            hipMemcpy(d_result, &result, sizeof(int), hipMemcpyHostToDevice);
            hipMemcpy(d_path, path.data(), V * sizeof(uint16_t), hipMemcpyHostToDevice);

            int blockSize = 256;
            int gridSize = (pos + blockSize - 1) / blockSize;

            hipLaunchKernelGGL(isSafeKernel, dim3(gridSize), dim3(blockSize), 0, 0, v, d_graph, d_path, pos, d_result, V);

            hipMemcpy(&result, d_result, sizeof(int), hipMemcpyDeviceToHost);

            if (result) {
                path[pos] = v;
                found = true;
                break;
            }
        }

        if (!found) {
            // No vertex can be added, backtrack
            return {};
        }
    }

    // Check if there's an edge from the last vertex to the first
    if (graph[path[V-1]][path[0]]) {
        return path;
    }

    // Free device memory
    hipFree(d_graph);
    hipFree(d_path);
    hipFree(d_result);

    return {};
}

int main() {
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
        std::cout << "No Hamiltonian cycle found." << std::endl;
    } else {
        std::cout << "Hamiltonian cycle: ";
        for (int v : result) {
            std::cout << v << " ";
        }
        std::cout << result[0] << std::endl;
    }

    return 0;
}