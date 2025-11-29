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

/* Include virtual_memory.h for TLS NUMA context integration */
extern "C" {
#include <randomx/virtual_memory.h>
}

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
    
    // Feature detection - exposed from virtual_memory
    static bool isOneGbPagesAvailable() { return vmem_isOneGbPagesAvailable(); }
    static bool isHugePagesAvailable() { return vmem_isHugePagesAvailable(); }

    // Set memory allocation policy for current thread
    static bool setMemoryPolicy(int node);
    
    // Restore default memory allocation policy
    static void restoreMemoryPolicy();
    
    // Get info about what the last allocation actually used
    static vmem_alloc_info_t getLastAllocInfo() { return vmem_getLastAllocInfo(); }
    
    // Print allocation info in human-readable form
    static void printAllocInfo(const vmem_alloc_info_t& info);
    
    /**
     * RAII helper for automatic NUMA policy management.
     * 
     * When constructed with a NUMA node:
     * 1. Sets the thread-local NUMA context in virtual_memory (vmem_setNumaNode)
     * 2. Sets the libnuma memory policy (numa_set_membind, etc.)
     * 
     * Any calls to allocLargePagesMemory() while this is in scope will:
     * - Automatically try 1GB pages first (if available and size >= 1GB)
     * - Fall back to 2MB pages
     * - Fall back to regular pages on the target NUMA node
     * - Apply madvise(MADV_RANDOM | MADV_WILLNEED) and mlock()
     * 
     * On destruction, both policies are restored to defaults.
     */
    class ScopedMemoryPolicy {
    private:
        bool need_restore;
        int target_node;
    public:
        ScopedMemoryPolicy(int node) : need_restore(false), target_node(node) {
            if (node >= 0) {
                // Set thread-local context for virtual_memory.c
                vmem_setNumaNode(node);
                // Set libnuma policy
                need_restore = NUMAOptimizer::setMemoryPolicy(node);
            }
        }
        ~ScopedMemoryPolicy() {
            // Always clear the TLS context
            vmem_clearNumaNode();
            // Restore libnuma policy if we changed it
            if (need_restore) {
                NUMAOptimizer::restoreMemoryPolicy();
            }
        }
        
        // Get what node this policy is targeting
        int getNode() const { return target_node; }
    };

private:
    static bool numa_initialized;
    static int memory_nodes;
    static int total_cpus;
    
    static void detectTopology();
};

#endif // NUMA_OPTIMIZER_H