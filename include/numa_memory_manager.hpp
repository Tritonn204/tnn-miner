#pragma once

#include "tnn-common.hpp"
#include "numa_optimizer.h"

template<typename Resource>
class NUMAMemoryManager {
public:
    struct NodeResource {
        int node_id;
        std::unique_ptr<Resource> resource;
    };

    using AllocatorFunc = std::function<std::unique_ptr<Resource>(int node_id)>;

    Resource* getResourceForThread(int thread_id) {
        if (!m_initialized) return nullptr;
        
        int node_id = -1;
        
        // Check if we've seen this thread before
        {
            std::lock_guard<std::mutex> lock(m_map_mutex);
            auto it = m_thread_to_node.find(thread_id);
            if (it != m_thread_to_node.end()) {
                node_id = it->second;
            }
        }
        
        // First time seeing this thread - detect its NUMA node
        if (node_id == -1) {
            #ifdef __linux__
            int cpu = sched_getcpu();
            if (cpu >= 0) {
                node_id = numa_node_of_cpu(cpu);
            }
            #elif defined(_WIN32)
            PROCESSOR_NUMBER proc_num = {0};
            proc_num.Group = 0xFFFF;
            proc_num.Number = 0xFF;
            GetCurrentProcessorNumberEx(&proc_num);

            if (proc_num.Group != 0xFFFF && proc_num.Number != 0xFF) {
                USHORT node_number = 0;
                if (GetNumaProcessorNodeEx(&proc_num, &node_number)) {
                    node_id = node_number;
                }
            }
            #endif
            
            if (node_id < 0 || node_id >= m_nodes.size()) {
                node_id = thread_id % m_nodes.size();  // Fallback
            }
            
            // Remember this thread's node
            {
                std::lock_guard<std::mutex> lock(m_map_mutex);
                m_thread_to_node[thread_id] = node_id;
            }
        }
        
        return m_nodes[node_id].resource.get();
    }
    
    // Simplified initialization - just create resources, no thread assignment
    bool initialize(int total_threads, AllocatorFunc allocator) {
        if (!lockThreads || !NUMAOptimizer::initialize()) {
            return false;
        }
        
        int numa_nodes = NUMAOptimizer::getMemoryNodes();
        if (numa_nodes <= 1) {
            return false;
        }
        
        m_nodes.resize(numa_nodes);
        
        // Just allocate resources per node
        for (int node = 0; node < numa_nodes; node++) {
            m_nodes[node].node_id = node;
            
            NUMAOptimizer::ScopedMemoryPolicy policy(node);
            m_nodes[node].resource = allocator(node);
            if (!m_nodes[node].resource) {
                return false;
            }
        }
        
        m_initialized = true;
        return true;
    }

    Resource* getResourceForNode(int node_id) {
        if (!m_initialized || node_id >= m_nodes.size()) return nullptr;
        return m_nodes[node_id].resource.get();
    }
    
    void forEachNode(std::function<void(int node_id, Resource*)> func) {
        for (auto& node : m_nodes) {
            func(node.node_id, node.resource.get());
        }
    }

private:
    std::vector<NodeResource> m_nodes;
    std::map<int, int> m_thread_to_node;
    std::mutex m_map_mutex;
    bool m_initialized = false;
};