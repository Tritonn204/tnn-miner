#ifndef NUMA_OPTIMIZER_H
#define NUMA_OPTIMIZER_H

#include <iostream>
#include <stdexcept>
#include "tnn-hugepages.hpp"

#ifdef __linux__
#include <numa.h>
#include <sched.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef TNN_RANDOMX
extern "C" {
#include <randomx/virtual_memory.hpp>
}
#endif

// ============================================================================
// Thread-local NUMA context
// ============================================================================

struct TnnAllocInfo {
    TnnPageType page_type;
    int numa_node;
    bool is_locked;
};

#ifdef __linux__
inline thread_local int tnn_tls_numa_node = -1;
inline thread_local TnnAllocInfo tnn_tls_last_alloc = {TNN_PAGE_REGULAR, -1, false};
#elif defined(_WIN32)
inline __declspec(thread) int tnn_tls_numa_node = -1;
inline __declspec(thread) TnnAllocInfo tnn_tls_last_alloc = {TNN_PAGE_REGULAR, -1, false};
#else
inline int tnn_tls_numa_node = -1;
inline TnnAllocInfo tnn_tls_last_alloc = {TNN_PAGE_REGULAR, -1, false};
#endif

inline void tnn_setNumaNode(int node) {
    tnn_tls_numa_node = node;
}

inline void tnn_clearNumaNode() {
    tnn_tls_numa_node = -1;
}

inline int tnn_getNumaNode() {
    return tnn_tls_numa_node;
}

inline TnnAllocInfo tnn_getLastAllocInfo() {
    return tnn_tls_last_alloc;
}

inline void tnn_setLastAllocInfo(TnnPageType type, int node, bool locked) {
    tnn_tls_last_alloc.page_type = type;
    tnn_tls_last_alloc.numa_node = node;
    tnn_tls_last_alloc.is_locked = locked;
}

// ============================================================================
// NUMAOptimizer class
// ============================================================================

class NUMAOptimizer {
public:
    struct NodeInfo {
        int node_id;
        int num_cpus;
        long memory_size_mb;
        bool has_memory;
    };

    static bool initialize();
    static int getMemoryNodes();
    static int getTotalCPUs();
    static void* allocateLocal(size_t size);
    static void* allocateOnNode(size_t size, int node);
    static void deallocate(void* ptr, size_t size);
    static void optimizeMemoryForMining(void* ptr, size_t size);
    static void printThreadBinding(int thread_id);
    static bool isAvailable();
    
    static const HugePagesInfo& getCachedHugePagesInfo() {
        static HugePagesInfo info = getHugePagesInfo();
        return info;
    }

    static bool isOneGbPagesAvailable() {
        const auto& info = getCachedHugePagesInfo();
        return info.page_size_1gb > 0 && info.free_1gb > 0;
    }

    static bool isHugePagesAvailable() {
        const auto& info = getCachedHugePagesInfo();
        return (info.page_size_2mb > 0 && info.free_2mb > 0) || 
               (info.page_size_1gb > 0 && info.free_1gb > 0);
    }

    static bool setMemoryPolicy(int node);
    static void restoreMemoryPolicy();
    
    static TnnAllocInfo getLastAllocInfo() { return tnn_getLastAllocInfo(); }
    static void printAllocInfo(const TnnAllocInfo& info);

    class ScopedMemoryPolicy {
    private:
        bool need_restore;
        int target_node;
    public:
        ScopedMemoryPolicy(int node) : need_restore(false), target_node(node) {
            if (node >= 0) {
                tnn_setNumaNode(node);
                need_restore = NUMAOptimizer::setMemoryPolicy(node);
            }
        }
        ~ScopedMemoryPolicy() {
            tnn_clearNumaNode();
            if (need_restore) {
                NUMAOptimizer::restoreMemoryPolicy();
            }
        }
        int getNode() const { return target_node; }
    };

private:
    static bool numa_initialized;
    static int memory_nodes;
    static int total_cpus;
    static void detectTopology();
};

#endif // NUMA_OPTIMIZER_H