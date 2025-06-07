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

    // Set memory allocation policy for current thread
    static bool setMemoryPolicy(int node);
    
    // Restore default memory allocation policy
    static void restoreMemoryPolicy();
    
    // RAII helper for automatic policy restoration
    class ScopedMemoryPolicy {
    private:
        bool need_restore;
    public:
        ScopedMemoryPolicy(int node) : need_restore(false) {
            need_restore = NUMAOptimizer::setMemoryPolicy(node);
        }
        ~ScopedMemoryPolicy() {
            if (need_restore) {
                NUMAOptimizer::restoreMemoryPolicy();
            }
        }
    };

private:
    static bool numa_initialized;
    static int memory_nodes;
    static int total_cpus;
    
    static void detectTopology();
};

#endif // NUMA_OPTIMIZER_H