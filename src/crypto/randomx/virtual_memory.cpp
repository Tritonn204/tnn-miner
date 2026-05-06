/*
Copyright (c) 2018-2019, tevador <tevador@gmail.com>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	* Neither the name of the copyright holder nor the
	  names of its contributors may be used to endorse or promote products
	  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* ============================================================================
 * Enhanced virtual memory allocator with:
 * - 1GB huge page support (Linux)
 * - NUMA-aware allocation via thread-local context
 * - madvise optimization (MADV_RANDOM | MADV_WILLNEED)
 * - mlock for preventing swap
 * - Automatic fallback chain: 1GB -> 2MB -> regular pages
 * ============================================================================ */

#include "numa_optimizer.hpp"
#include "virtual_memory.hpp"

#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#define _GNU_SOURCE 1
#ifdef __APPLE__
#include <mach/vm_statistics.h>
#include <TargetConditionals.h>
#include <AvailabilityMacros.h>
# if TARGET_OS_OSX
#  define USE_PTHREAD_JIT_WP 1
#  include <pthread.h>
#  include <sys/utsname.h>
#  include <stdio.h>
# endif
#endif
#include <sys/types.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

/* Linux NUMA support */
#if defined(__linux__)
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <fcntl.h>

/* Huge page defines */
#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_MASK
#define MAP_HUGE_MASK 0x3f
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

#define SIZE_2MB (2ULL << 20)
#define SIZE_1GB (1ULL << 30)

#endif /* __linux__ */

#define PAGE_READONLY PROT_READ
#define PAGE_READWRITE (PROT_READ | PROT_WRITE)
#define PAGE_EXECUTE_READ (PROT_READ | PROT_EXEC)
#define PAGE_EXECUTE_READWRITE (PROT_READ | PROT_WRITE | PROT_EXEC)
#endif /* !_WIN32 */

/* ============================================================================
 * API compatibility shims - map old vmem_* API to new tnn_* API
 * ============================================================================ */

void vmem_setNumaNode(int node) {
    tnn_setNumaNode(node);
}

void vmem_clearNumaNode(void) {
    tnn_clearNumaNode();
}

int vmem_getNumaNode(void) {
    return tnn_getNumaNode();
}

vmem_alloc_info_t vmem_getLastAllocInfo(void) {
    TnnAllocInfo tnn_info = tnn_getLastAllocInfo();
    vmem_alloc_info_t info;
    
    // Map TnnPageType to vmem_page_type_t
    switch (tnn_info.page_type) {
        case TNN_PAGE_1GB:    info.page_type = VMEM_PAGE_1GB; break;
        case TNN_PAGE_2MB:    info.page_type = VMEM_PAGE_2MB; break;
        default:              info.page_type = VMEM_PAGE_REGULAR; break;
    }
    info.numa_node = tnn_info.numa_node;
    info.is_locked = tnn_info.is_locked;
    
    return info;
}

/* ============================================================================
 * macOS version checking (for JIT write protect)
 * ============================================================================ */

#if defined(USE_PTHREAD_JIT_WP) && defined(MAC_OS_VERSION_11_0) \
    && MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_11_0
static int MacOSchecked, MacOSver;

static int32_t __isOSVersionAtLeast(int32_t major, int32_t minor, int32_t subminor) {
    if (!MacOSchecked) {
        struct utsname ut;
        int mmaj, mmin;
        uname(&ut);
        sscanf(ut.release, "%d.%d", &mmaj, &mmin);
        mmaj -= 9;
        MacOSver = (mmaj << 8) | mmin;
        MacOSchecked = 1;
    }
    return MacOSver >= ((major << 8) | minor);
}
#endif

/* ============================================================================
 * Windows privilege handling
 * ============================================================================ */

#if defined(_WIN32) || defined(__CYGWIN__)
#define Fail(func) do { *errfunc = func; return GetLastError(); } while(0)

int setPrivilege(const char* pszPrivilege, BOOL bEnable, char **errfunc) {
    HANDLE           hToken;
    TOKEN_PRIVILEGES tp;
    BOOL             status;
    DWORD            error = 0;

    *errfunc = NULL;

    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken))
        Fail("OpenProcessToken");

    if (!LookupPrivilegeValue(NULL, pszPrivilege, &tp.Privileges[0].Luid)) {
        *errfunc = "LookupPrivilegeValue";
        error = GetLastError();
        goto out;
    }

    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = bEnable ? SE_PRIVILEGE_ENABLED : 0;

    status = AdjustTokenPrivileges(hToken, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0);
    error = GetLastError();
    if (!status || (error != ERROR_SUCCESS)) {
        *errfunc = "AdjustTokenPrivileges";
        goto out;
    }

out:
    if (!CloseHandle(hToken)) {
        if (*errfunc == NULL) {
            *errfunc = "CloseHandle";
            error = GetLastError();
        }
    }
    return error;
}
#else
#define Fail(func) do { *errfunc = func; return errno; } while(0)
#endif

/* ============================================================================
 * Memory optimization (madvise + mlock)
 * ============================================================================ */

#if defined(__linux__)
static void optimizeMemory(void* ptr, size_t size) {
    if (!ptr || size == 0) return;
    
    /* Tell kernel about access pattern - RandomX does random reads */
    madvise(ptr, size, MADV_WILLNEED);
    madvise(ptr, size, MADV_RANDOM);
    
    /* Try to lock in RAM to prevent swapping */
    TnnAllocInfo info = tnn_getLastAllocInfo();
    if (mlock(ptr, size) == 0) {
        tnn_setLastAllocInfo(info.page_type, info.numa_node, true);
    }
}
#elif defined(_WIN32)
static void optimizeMemory(void* ptr, size_t size) {
    if (!ptr || size == 0) return;
    
    /* VirtualLock to prevent paging */
    TnnAllocInfo info = tnn_getLastAllocInfo();
    if (VirtualLock(ptr, size)) {
        tnn_setLastAllocInfo(info.page_type, info.numa_node, true);
    }
}
#else
static void optimizeMemory(void* ptr, size_t size) {
    (void)ptr; (void)size;
}
#endif

/* ============================================================================
 * Linux NUMA-aware huge page allocation
 * ============================================================================ */

#if defined(__linux__)

/* Set NUMA memory policy for allocation */
static void setNumaPolicy(int node) {
    if (node < 0 || numa_available() < 0) return;
    
    unsigned long nodemask = 1UL << node;
    set_mempolicy(MPOL_BIND, &nodemask, sizeof(nodemask) * 8);
}

/* Restore default NUMA policy */
static void clearNumaPolicy(void) {
    if (numa_available() < 0) return;
    set_mempolicy(MPOL_DEFAULT, NULL, 0);
}

/* Allocate with full fallback chain: 1GB -> 2MB -> regular, all NUMA-aware */
static void* allocLargePagesLinux(size_t bytes) {
    void *mem = MAP_FAILED;
    int node = tnn_getNumaNode();
    
    /* Reset allocation info */
    tnn_setLastAllocInfo(TNN_PAGE_REGULAR, node, false);
    
    /* Set NUMA policy if node specified */
    if (node >= 0) {
        setNumaPolicy(node);
    }
    
    /* 
     * Fallback chain:
     * 1. Try 1GB pages for large allocations (dataset ~2GB)
     * 2. Try 2MB pages 
     * 3. Fall back to regular pages with NUMA binding
     */
    
    /* Try 1GB pages first for large allocations */
    if (bytes >= SIZE_1GB && NUMAOptimizer::isOneGbPagesAvailable()) {
        size_t aligned = alignSize(bytes, SIZE_1GB);
        mem = mmap(NULL, aligned, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB | MAP_POPULATE,
            -1, 0);
        if (mem != MAP_FAILED) {
            tnn_setLastAllocInfo(TNN_PAGE_1GB, node, false);
            goto done;
        }
    }
    
    /* Try 2MB pages */
    if (NUMAOptimizer::isHugePagesAvailable()) {
        mem = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB | MAP_POPULATE,
            -1, 0);
        if (mem != MAP_FAILED) {
            tnn_setLastAllocInfo(TNN_PAGE_2MB, node, false);
            goto done;
        }
        
        /* Try without explicit size hint (uses system default huge page size) */
        mem = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE,
            -1, 0);
        if (mem != MAP_FAILED) {
            tnn_setLastAllocInfo(TNN_PAGE_2MB, node, false);
            goto done;
        }
    }
    
    /* Fall back to regular pages */
    if (node >= 0 && numa_available() >= 0) {
        /* Use numa_alloc_onnode for NUMA-local regular pages */
        mem = numa_alloc_onnode(bytes, node);
        if (mem != NULL) {
            /* numa_alloc returns NULL on failure, not MAP_FAILED */
            tnn_setLastAllocInfo(TNN_PAGE_REGULAR, node, false);
            /* Clear policy before returning */
            clearNumaPolicy();
            optimizeMemory(mem, bytes);
            return mem;
        }
        mem = MAP_FAILED;  /* Normalize for done: check */
    }
    
    /* Last resort: regular mmap */
    mem = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
        -1, 0);

done:
    /* Restore default NUMA policy */
    if (node >= 0) {
        clearNumaPolicy();
    }
    
    if (mem == MAP_FAILED) {
        return NULL;
    }
    
    optimizeMemory(mem, bytes);
    return mem;
}

#endif /* __linux__ */

/* ============================================================================
 * Public API - allocMemoryPages (regular page allocation)
 * ============================================================================ */

void* allocMemoryPages(size_t bytes) {
    void* mem;
#if defined(_WIN32) || defined(__CYGWIN__)
    mem = VirtualAlloc(NULL, bytes, MEM_COMMIT, PAGE_READWRITE);
#else
    #if defined(__NetBSD__)
        #define RESERVED_FLAGS PROT_MPROTECT(PROT_EXEC)
    #else
        #define RESERVED_FLAGS 0
    #endif
    #ifdef USE_PTHREAD_JIT_WP
        #define MEXTRA MAP_JIT
        #define PEXTRA PROT_EXEC
    #else
        #define MEXTRA 0
        #define PEXTRA 0
    #endif
    mem = mmap(NULL, bytes, PAGE_READWRITE | RESERVED_FLAGS | PEXTRA, MAP_ANONYMOUS | MAP_PRIVATE | MEXTRA, -1, 0);
    if (mem == MAP_FAILED)
        mem = NULL;
#if defined(USE_PTHREAD_JIT_WP) && defined(MAC_OS_VERSION_11_0) \
    && MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_11_0
    if (__builtin_available(macOS 11.0, *)) {
        pthread_jit_write_protect_np(0);
    }
#endif
#endif
    return mem;
}

/* ============================================================================
 * Public API - page protection
 * ============================================================================ */

static inline int pageProtect(void* ptr, size_t bytes, int rules, char **errfunc) {
#if defined(_WIN32) || defined(__CYGWIN__)
    DWORD oldp;
    if (!VirtualProtect(ptr, bytes, (DWORD)rules, &oldp)) {
        Fail("VirtualProtect");
    }
#else
    if (-1 == mprotect(ptr, bytes, rules))
        Fail("mprotect");
#endif
    return 0;
}

void setPagesRW(void* ptr, size_t bytes) {
    char *errfunc;
#if defined(USE_PTHREAD_JIT_WP) && defined(MAC_OS_VERSION_11_0) \
    && MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_11_0
    if (__builtin_available(macOS 11.0, *)) {
        pthread_jit_write_protect_np(0);
    } else {
        pageProtect(ptr, bytes, PAGE_READWRITE, &errfunc);
    }
#else
    pageProtect(ptr, bytes, PAGE_READWRITE, &errfunc);
#endif
}

void setPagesRX(void* ptr, size_t bytes) {
    char *errfunc;
#if defined(USE_PTHREAD_JIT_WP) && defined(MAC_OS_VERSION_11_0) \
    && MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_11_0
    if (__builtin_available(macOS 11.0, *)) {
        pthread_jit_write_protect_np(1);
        __builtin___clear_cache((char*)ptr, ((char*)ptr) + bytes);
    } else {
        pageProtect(ptr, bytes, PAGE_EXECUTE_READ, &errfunc);
    }
#else
    pageProtect(ptr, bytes, PAGE_EXECUTE_READ, &errfunc);
#endif
}

void setPagesRWX(void* ptr, size_t bytes) {
    char *errfunc;
    pageProtect(ptr, bytes, PAGE_EXECUTE_READWRITE, &errfunc);
}

/* ============================================================================
 * Public API - allocLargePagesMemory (main entry point)
 * ============================================================================ */

extern "C" {

void* allocLargePagesMemory(size_t bytes) {
    void* mem;
    char *errfunc;

#if defined(_WIN32) || defined(__CYGWIN__)
    /* Windows large page allocation */
    if (setPrivilege("SeLockMemoryPrivilege", 1, &errfunc))
        return NULL;
    size_t pageMinimum = GetLargePageMinimum();
    if (!pageMinimum) {
        return NULL;
    }
    
    int node = tnn_getNumaNode();
    tnn_setLastAllocInfo(TNN_PAGE_2MB, node, false);
    
    if (node >= 0) {
        /* NUMA-aware allocation on Windows */
        mem = VirtualAllocExNuma(
            GetCurrentProcess(),
            NULL,
            alignSize(bytes, pageMinimum),
            MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
            PAGE_READWRITE,
            (UCHAR)node
        );
    } else {
        mem = VirtualAlloc(NULL, alignSize(bytes, pageMinimum), 
            MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
    }
    
    if (mem) {
        optimizeMemory(mem, bytes);
    }

#elif defined(__APPLE__)
    /* macOS superpage allocation */
    tnn_setLastAllocInfo(TNN_PAGE_2MB, -1, false);
    
    mem = mmap(NULL, bytes, PROT_READ | PROT_WRITE, 
        MAP_PRIVATE | MAP_ANON, VM_FLAGS_SUPERPAGE_SIZE_2MB, 0);
    if (mem == MAP_FAILED)
        mem = NULL;

#elif defined(__FreeBSD__)
    /* FreeBSD superpage allocation */
    tnn_setLastAllocInfo(TNN_PAGE_2MB, -1, false);
    
    mem = mmap(NULL, bytes, PROT_READ | PROT_WRITE, 
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_ALIGNED_SUPER, -1, 0);
    if (mem == MAP_FAILED)
        mem = NULL;

#elif defined(__linux__)
    /* Linux: Full NUMA-aware allocation with fallback chain */
    mem = allocLargePagesLinux(bytes);

#elif defined(__OpenBSD__) || defined(__NetBSD__)
    /* OpenBSD/NetBSD: No huge page support */
    tnn_setLastAllocInfo(TNN_PAGE_REGULAR, -1, false);
    mem = NULL;

#else
    /* Unknown platform */
    tnn_setLastAllocInfo(TNN_PAGE_REGULAR, -1, false);
    mem = NULL;
#endif

    return mem;
}

/* ============================================================================
 * Public API - freePagedMemory
 * ============================================================================ */

void freePagedMemory(void* ptr, size_t bytes) {
#if defined(_WIN32) || defined(__CYGWIN__)
    VirtualFree(ptr, 0, MEM_RELEASE);
#elif defined(__linux__)
    munmap(ptr, bytes);
#else
    munmap(ptr, bytes);
#endif
}

}