#ifndef NUMA_OPTIMIZER_H
#define NUMA_OPTIMIZER_H

#include <iostream>
#include <stdexcept>

#ifdef __linux__
#include <numa.h>
#include <sched.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

class NUMAOptimizer {
public:
    struct NodeInfo {
        int node_id;
        int num_cpus;
        long memory_size_mb;
        bool has_memory;
    };

    // Initialize NUMA system and return topology info
    static bool initialize();
    
    // Get number of NUMA nodes with memory
    static int getMemoryNodes();
    
    // Get total number of CPUs
    static int getTotalCPUs();
    
    // Bind thread to specific NUMA node (1 thread per node strategy)
    static bool bindThreadToNode(int thread_id, int total_threads);
    
    // Allocate memory on current thread's NUMA node
    static void* allocateLocal(size_t size);
    
    // Allocate memory on specific NUMA node
    static void* allocateOnNode(size_t size, int node);
    
    // Free NUMA-allocated memory
    static void deallocate(void* ptr, size_t size);
    
    // Configure memory for mining workloads
    static void optimizeMemoryForMining(void* ptr, size_t size);
    
    // Print current thread's NUMA binding
    static void printThreadBinding(int thread_id);
    
    // Check if NUMA is available on this system
    static bool isAvailable();

private:
    static bool numa_initialized;
    static int memory_nodes;
    static int total_cpus;
    
    static void detectTopology();
};

#endif // NUMA_OPTIMIZER_H