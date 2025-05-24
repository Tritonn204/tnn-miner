#include "numa_optimizer.h"
#include <vector>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#include <memoryapi.h>
#include <processtopologyapi.h>
#include <processthreadsapi.h>
#include <sysinfoapi.h>

// Define minimum Windows version for NUMA support
#ifndef _WIN32_WINNT_WIN7
#define _WIN32_WINNT_WIN7 0x0601
#endif
#ifndef _WIN32_WINNT_WIN8
#define _WIN32_WINNT_WIN8 0x0602
#endif

// Check if we have newer APIs available
#if _WIN32_WINNT >= _WIN32_WINNT_WIN8
#define HAVE_WIN8_NUMA_API 1
#else
#define HAVE_WIN8_NUMA_API 0
#endif

#pragma comment(lib, "kernel32.lib")

// Function pointer types for dynamic loading of newer APIs
typedef BOOL (WINAPI *PFN_GetNumaHighestNodeNumber)(PULONG HighestNodeNumber);
typedef BOOL (WINAPI *PFN_GetNumaNodeProcessorMaskEx)(USHORT Node, PGROUP_AFFINITY ProcessorMask);
typedef LPVOID (WINAPI *PFN_VirtualAllocExNuma)(HANDLE hProcess, LPVOID lpAddress, SIZE_T dwSize, DWORD flAllocationType, DWORD flProtect, DWORD nndPreferred);
typedef BOOL (WINAPI *PFN_GetCurrentProcessorNumberEx)(PPROCESSOR_NUMBER ProcNumber);
typedef BOOL (WINAPI *PFN_GetNumaProcessorNodeEx)(PPROCESSOR_NUMBER Processor, PUSHORT NodeNumber);
typedef BOOL (WINAPI *PFN_SetThreadGroupAffinity)(HANDLE hThread, CONST GROUP_AFFINITY* GroupAffinity, PGROUP_AFFINITY PreviousGroupAffinity);

// Global function pointers
static PFN_GetNumaHighestNodeNumber g_GetNumaHighestNodeNumber = nullptr;
static PFN_GetNumaNodeProcessorMaskEx g_GetNumaNodeProcessorMaskEx = nullptr;
static PFN_VirtualAllocExNuma g_VirtualAllocExNuma = nullptr;
static PFN_GetCurrentProcessorNumberEx g_GetCurrentProcessorNumberEx = nullptr;
static PFN_GetNumaProcessorNodeEx g_GetNumaProcessorNodeEx = nullptr;
static PFN_SetThreadGroupAffinity g_SetThreadGroupAffinity = nullptr;

static bool g_numa_functions_loaded = false;

static void LoadNumaFunctions() {
    if (g_numa_functions_loaded) return;
    
    HMODULE hKernel = GetModuleHandleA("kernel32.dll");
    if (!hKernel) return;
    
    g_GetNumaHighestNodeNumber = (PFN_GetNumaHighestNodeNumber)GetProcAddress(hKernel, "GetNumaHighestNodeNumber");
    g_GetNumaNodeProcessorMaskEx = (PFN_GetNumaNodeProcessorMaskEx)GetProcAddress(hKernel, "GetNumaNodeProcessorMaskEx");
    g_VirtualAllocExNuma = (PFN_VirtualAllocExNuma)GetProcAddress(hKernel, "VirtualAllocExNuma");
    g_GetCurrentProcessorNumberEx = (PFN_GetCurrentProcessorNumberEx)GetProcAddress(hKernel, "GetCurrentProcessorNumberEx");
    g_GetNumaProcessorNodeEx = (PFN_GetNumaProcessorNodeEx)GetProcAddress(hKernel, "GetNumaProcessorNodeEx");
    g_SetThreadGroupAffinity = (PFN_SetThreadGroupAffinity)GetProcAddress(hKernel, "SetThreadGroupAffinity");
    
    g_numa_functions_loaded = true;
}

#endif

bool NUMAOptimizer::numa_initialized = false;
int NUMAOptimizer::memory_nodes = 0;
int NUMAOptimizer::total_cpus = 0;

bool NUMAOptimizer::initialize() {
#ifdef __linux__
    if (numa_available() < 0) {
        std::cerr << "NUMA not available on this system" << std::endl;
        return false;
    }
    
    detectTopology();
    numa_initialized = true;
    
    std::cout << "NUMA initialized: " << memory_nodes << " memory nodes, " 
              << total_cpus << " CPUs total" << std::endl;
    
    // Print topology
    for (int node = 0; node < memory_nodes; node++) {
        if (numa_node_size(node, nullptr) > 0) {
            std::cout << "Node " << node << ": " 
                      << (numa_node_size(node, nullptr) / (1024*1024)) << "MB RAM, CPUs: ";
            
            struct bitmask* cpus = numa_allocate_cpumask();
            numa_node_to_cpus(node, cpus);
            for (int cpu = 0; cpu < total_cpus; cpu++) {
                if (numa_bitmask_isbitset(cpus, cpu)) {
                    std::cout << cpu << " ";
                }
            }
            std::cout << std::endl;
            numa_free_cpumask(cpus);
        }
    }
    
    return true;
    
#elif defined(_WIN32)
    LoadNumaFunctions();
    
    if (!g_GetNumaHighestNodeNumber) {
        std::cerr << "NUMA functions not available on this Windows version" << std::endl;
        return false;
    }
    
    ULONG highest_node_number;
    if (!g_GetNumaHighestNodeNumber(&highest_node_number)) {
        std::cerr << "Failed to get NUMA node count" << std::endl;
        return false;
    }
    
    detectTopology();
    numa_initialized = true;
    
    std::cout << "NUMA initialized (Windows): " << memory_nodes << " memory nodes, " 
              << total_cpus << " CPUs total" << std::endl;
    
    // Print Windows NUMA topology
    if (g_GetNumaNodeProcessorMaskEx) {
        for (UCHAR node = 0; node <= highest_node_number; node++) {
            GROUP_AFFINITY processor_mask;
            if (g_GetNumaNodeProcessorMaskEx(node, &processor_mask)) {
                std::cout << "Node " << (int)node << ": CPUs in group " 
                          << processor_mask.Group << ": ";
                
                for (int i = 0; i < 64; i++) {
                    if (processor_mask.Mask & (1ULL << i)) {
                        std::cout << i << " ";
                    }
                }
                std::cout << std::endl;
            }
        }
    }
    
    return true;
#else
    std::cerr << "NUMA optimization only available on Linux and Windows" << std::endl;
    return false;
#endif
}

void NUMAOptimizer::detectTopology() {
#ifdef __linux__
    total_cpus = numa_num_configured_cpus();
    int max_nodes = numa_max_node() + 1;
    
    memory_nodes = 0;
    for (int node = 0; node < max_nodes; node++) {
        if (numa_node_size(node, nullptr) > 0) {
            memory_nodes++;
        }
    }
    
#elif defined(_WIN32)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    total_cpus = sysinfo.dwNumberOfProcessors;
    
    if (g_GetNumaHighestNodeNumber) {
        ULONG highest_node_number;
        if (g_GetNumaHighestNodeNumber(&highest_node_number)) {
            memory_nodes = highest_node_number + 1;
        } else {
            memory_nodes = 1;
        }
    } else {
        memory_nodes = 1;
    }
#endif
}

int NUMAOptimizer::getMemoryNodes() {
    return memory_nodes;
}

int NUMAOptimizer::getTotalCPUs() {
    return total_cpus;
}

bool NUMAOptimizer::bindThreadToNode(int thread_id, int total_threads) {
#ifdef __linux__
    if (!numa_initialized) {
        std::cerr << "NUMA not initialized" << std::endl;
        return false;
    }
    
    // Strategy: 1 thread per NUMA node, cycle if more threads than nodes
    int target_node = thread_id % memory_nodes;
    
    // Ensure the node has memory
    while (numa_node_size(target_node, nullptr) <= 0) {
        target_node = (target_node + 1) % memory_nodes;
    }
    
    // Bind to NUMA node for both CPU and memory
    numa_run_on_node(target_node);
    numa_set_localalloc();
    
    // Get CPUs for this NUMA node and bind to first available
    struct bitmask* cpus = numa_allocate_cpumask();
    numa_node_to_cpus(target_node, cpus);
    
    // Find first CPU on this node
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    for (int cpu = 0; cpu < total_cpus; cpu++) {
        if (numa_bitmask_isbitset(cpus, cpu)) {
            CPU_SET(cpu, &cpuset);
            break; // Bind to first CPU on the node
        }
    }
    
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "Failed to set CPU affinity for thread " << thread_id << std::endl;
        numa_free_cpumask(cpus);
        return false;
    }
    
    numa_free_cpumask(cpus);
    
    std::cout << "Thread " << thread_id << " bound to NUMA node " << target_node << std::endl;
    return true;
    
#elif defined(_WIN32)
    if (!numa_initialized || !g_GetNumaNodeProcessorMaskEx || !g_SetThreadGroupAffinity) {
        std::cerr << "NUMA binding not available on this Windows version" << std::endl;
        return false;
    }
    
    UCHAR target_node = thread_id % memory_nodes;
    
    // Get processor mask for this NUMA node
    GROUP_AFFINITY processor_mask;
    if (!g_GetNumaNodeProcessorMaskEx(target_node, &processor_mask)) {
        std::cerr << "Failed to get processor mask for NUMA node " << (int)target_node << std::endl;
        return false;
    }
    
    // Set thread affinity to this NUMA node
    if (!g_SetThreadGroupAffinity(GetCurrentThread(), &processor_mask, nullptr)) {
        std::cerr << "Failed to set thread affinity for thread " << thread_id << std::endl;
        return false;
    }
    
    // Set ideal processor (first CPU in the node)
    PROCESSOR_NUMBER ideal_proc = {0};
    ideal_proc.Group = processor_mask.Group;
    ideal_proc.Number = 0;
    
    // Find first set bit in mask
    for (BYTE i = 0; i < 64; i++) {
        if (processor_mask.Mask & (1ULL << i)) {
            ideal_proc.Number = i;
            break;
        }
    }
    
    SetThreadIdealProcessorEx(GetCurrentThread(), &ideal_proc, nullptr);
    
    return true;
#else
    return false;
#endif
}

void* NUMAOptimizer::allocateLocal(size_t size) {
#ifdef __linux__
    if (!numa_initialized) {
        return malloc(size);
    }
    return numa_alloc_local(size);
    
#elif defined(_WIN32)
    if (!numa_initialized || !g_VirtualAllocExNuma || !g_GetCurrentProcessorNumberEx || !g_GetNumaProcessorNodeEx) {
        return malloc(size);
    }
    
    // Get current processor's NUMA node
    PROCESSOR_NUMBER proc_num;
    if (!g_GetCurrentProcessorNumberEx(&proc_num)) {
        return malloc(size);
    }
    
    USHORT node_number;
    if (!g_GetNumaProcessorNodeEx(&proc_num, &node_number)) {
        return malloc(size);
    }
    
    // Allocate on current NUMA node
    void* ptr = g_VirtualAllocExNuma(
        GetCurrentProcess(),
        NULL,
        size,
        MEM_COMMIT | MEM_RESERVE,
        PAGE_READWRITE,
        node_number
    );
    
    if (!ptr) {
        // Fallback to regular allocation
        ptr = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    }
    
    return ptr;
#else
    return malloc(size);
#endif
}

void* NUMAOptimizer::allocateOnNode(size_t size, int node) {
#ifdef __linux__
    if (!numa_initialized) {
        return malloc(size);
    }
    return numa_alloc_onnode(size, node);
    
#elif defined(_WIN32)
    if (!numa_initialized || !g_VirtualAllocExNuma) {
        return malloc(size);
    }
    
    void* ptr = g_VirtualAllocExNuma(
        GetCurrentProcess(),
        NULL,
        size,
        MEM_COMMIT | MEM_RESERVE,
        PAGE_READWRITE,
        (UCHAR)node
    );
    
    if (!ptr) {
        // Fallback to regular allocation
        ptr = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    }
    
    return ptr;
#else
    return malloc(size);
#endif
}

void NUMAOptimizer::deallocate(void* ptr, size_t size) {
#ifdef __linux__
    if (!numa_initialized) {
        free(ptr);
    } else {
        numa_free(ptr, size);
    }
    
#elif defined(_WIN32)
    if (!numa_initialized || !ptr) {
        free(ptr);
    } else {
        VirtualFree(ptr, 0, MEM_RELEASE);
    }
#else
    free(ptr);
#endif
}

void NUMAOptimizer::optimizeMemoryForMining(void* ptr, size_t size) {
#ifdef __linux__
    if (!ptr || size == 0) return;
    
    // Use huge pages for large allocations (reduces TLB pressure)
    if (size >= 2 * 1024 * 1024) {  // >= 2MB
        madvise(ptr, size, MADV_HUGEPAGE);
    }
    
    // Tell kernel we'll access this memory soon
    madvise(ptr, size, MADV_WILLNEED);
    
    // For mining: sequential access pattern
    madvise(ptr, size, MADV_SEQUENTIAL);
    
    // Touch memory to ensure it's allocated locally
    volatile char* touch = (volatile char*)ptr;
    for (size_t i = 0; i < size; i += 4096) {
        touch[i] = touch[i];
    }
    
#elif defined(_WIN32)
    if (!ptr || size == 0) return;
    
    // Enable large pages if possible (requires SeLockMemoryPrivilege)
    SIZE_T min_large_page_size = GetLargePageMinimum();
    if (min_large_page_size > 0 && size >= min_large_page_size) {
        // Try to enable large page support
        HANDLE hToken;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken)) {
            TOKEN_PRIVILEGES tp = {0};
            tp.PrivilegeCount = 1;
            tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
            
            if (LookupPrivilegeValueA(NULL, "SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) {
                AdjustTokenPrivileges(hToken, FALSE, &tp, 0, NULL, 0);
            }
            CloseHandle(hToken);
        }
    }
    
    // Touch memory to ensure it's committed and local
    volatile char* touch = (volatile char*)ptr;
    for (size_t i = 0; i < size; i += 4096) {
        touch[i] = touch[i];
    }
#endif
}

void NUMAOptimizer::printThreadBinding(int thread_id) {
#ifdef __linux__
    int current_node = numa_node_of_cpu(sched_getcpu());
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0) {
        std::cout << "Thread " << thread_id << " running on NUMA node " << current_node 
                  << ", CPU(s): ";
        for (int cpu = 0; cpu < total_cpus; cpu++) {
            if (CPU_ISSET(cpu, &cpuset)) {
                std::cout << cpu << " ";
            }
        }
        std::cout << std::endl;
    }
    
#elif defined(_WIN32)
    if (g_GetCurrentProcessorNumberEx && g_GetNumaProcessorNodeEx) {
        PROCESSOR_NUMBER proc_num;
        if (g_GetCurrentProcessorNumberEx(&proc_num)) {
            USHORT node_number;
            if (g_GetNumaProcessorNodeEx(&proc_num, &node_number)) {
                std::cout << "Thread " << thread_id << " running on NUMA node " << node_number
                          << ", Processor Group " << proc_num.Group 
                          << ", CPU " << proc_num.Number << std::endl;
            }
        }
    } else {
        std::cout << "Thread " << thread_id << " NUMA info not available on this Windows version" << std::endl;
    }
#endif
}

bool NUMAOptimizer::isAvailable() {
#ifdef __linux__
    return numa_available() >= 0;
#elif defined(_WIN32)
    LoadNumaFunctions();
    if (g_GetNumaHighestNodeNumber) {
        ULONG highest_node_number;
        return g_GetNumaHighestNodeNumber(&highest_node_number) && highest_node_number > 0;
    }
    return false;
#else
    return false;
#endif
}