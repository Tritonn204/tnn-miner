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

#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#define _GNU_SOURCE	1
#ifdef __APPLE__
#include <mach/vm_statistics.h>
#include <TargetConditionals.h>
#include <AvailabilityMacros.h>
# if TARGET_OS_OSX
#  define USE_PTHREAD_JIT_WP	1
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

/* Check for 1GB page availability at runtime */
static int g_1gb_pages_available = -1;  /* -1 = not checked, 0 = no, 1 = yes */
static int g_2mb_pages_available = -1;

static int check_hugepages_available(size_t page_size) {
	char path[128];
	int fd;
	char buf[32];
	ssize_t n;
	unsigned long count = 0;
	
	/* Check both global and per-node paths */
	if (page_size == SIZE_1GB) {
		snprintf(path, sizeof(path), "/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages");
	} else {
		snprintf(path, sizeof(path), "/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages");
	}
	
	fd = open(path, O_RDONLY);
	if (fd < 0) return 0;
	
	n = read(fd, buf, sizeof(buf) - 1);
	close(fd);
	
	if (n > 0) {
		buf[n] = '\0';
		count = strtoul(buf, NULL, 10);
	}
	
	return count > 0 ? 1 : 0;
}

#endif /* __linux__ */

#define PAGE_READONLY PROT_READ
#define PAGE_READWRITE (PROT_READ | PROT_WRITE)
#define PAGE_EXECUTE_READ (PROT_READ | PROT_EXEC)
#define PAGE_EXECUTE_READWRITE (PROT_READ | PROT_WRITE | PROT_EXEC)
#endif /* !_WIN32 */

#include "virtual_memory.h"

/* ============================================================================
 * Thread-local NUMA context
 * ============================================================================ */

#if defined(__linux__)
static __thread int tls_numa_node = -1;
static __thread vmem_alloc_info_t tls_last_alloc = {VMEM_PAGE_REGULAR, -1, false};
#elif defined(_WIN32)
static __declspec(thread) int tls_numa_node = -1;
static __declspec(thread) vmem_alloc_info_t tls_last_alloc = {VMEM_PAGE_REGULAR, -1, false};
#else
/* Fallback for systems without TLS - not truly thread-safe but functional */
static int tls_numa_node = -1;
static vmem_alloc_info_t tls_last_alloc = {VMEM_PAGE_REGULAR, -1, false};
#endif

void vmem_setNumaNode(int node) {
	tls_numa_node = node;
}

void vmem_clearNumaNode(void) {
	tls_numa_node = -1;
}

int vmem_getNumaNode(void) {
	return tls_numa_node;
}

vmem_alloc_info_t vmem_getLastAllocInfo(void) {
	return tls_last_alloc;
}

/* ============================================================================
 * Feature detection
 * ============================================================================ */

bool vmem_isOneGbPagesAvailable(void) {
#if defined(__linux__)
	if (g_1gb_pages_available < 0) {
		g_1gb_pages_available = check_hugepages_available(SIZE_1GB);
	}
	return g_1gb_pages_available > 0;
#else
	return false;
#endif
}

bool vmem_isHugePagesAvailable(void) {
#if defined(__linux__)
	if (g_2mb_pages_available < 0) {
		g_2mb_pages_available = check_hugepages_available(SIZE_2MB);
	}
	return g_2mb_pages_available > 0;
#elif defined(__APPLE__) || defined(__FreeBSD__)
	return true;  /* Superpage support */
#elif defined(_WIN32)
	return GetLargePageMinimum() > 0;
#else
	return false;
#endif
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
#define Fail(func)	do  {*errfunc = func; return GetLastError();} while(0)

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
#define Fail(func)	do  {*errfunc = func; return errno;} while(0)
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
	if (mlock(ptr, size) == 0) {
		tls_last_alloc.is_locked = true;
	}
}
#elif defined(_WIN32)
static void optimizeMemory(void* ptr, size_t size) {
	if (!ptr || size == 0) return;
	
	/* VirtualLock to prevent paging */
	if (VirtualLock(ptr, size)) {
		tls_last_alloc.is_locked = true;
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
	int node = tls_numa_node;
	
	/* Reset allocation info */
	tls_last_alloc.page_type = VMEM_PAGE_REGULAR;
	tls_last_alloc.numa_node = node;
	tls_last_alloc.is_locked = false;
	
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
	if (bytes >= SIZE_1GB && vmem_isOneGbPagesAvailable()) {
		size_t aligned = alignSize(bytes, SIZE_1GB);
		mem = mmap(NULL, aligned, PROT_READ | PROT_WRITE,
			MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB | MAP_POPULATE,
			-1, 0);
		if (mem != MAP_FAILED) {
			tls_last_alloc.page_type = VMEM_PAGE_1GB;
			goto done;
		}
	}
	
	/* Try 2MB pages */
	if (vmem_isHugePagesAvailable()) {
		mem = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
			MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB | MAP_POPULATE,
			-1, 0);
		if (mem != MAP_FAILED) {
			tls_last_alloc.page_type = VMEM_PAGE_2MB;
			goto done;
		}
		
		/* Try without explicit size hint (uses system default huge page size) */
		mem = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
			MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE,
			-1, 0);
		if (mem != MAP_FAILED) {
			tls_last_alloc.page_type = VMEM_PAGE_2MB;
			goto done;
		}
	}
	
	/* Fall back to regular pages */
	if (node >= 0 && numa_available() >= 0) {
		/* Use numa_alloc_onnode for NUMA-local regular pages */
		mem = numa_alloc_onnode(bytes, node);
		if (mem != NULL) {
			/* numa_alloc returns NULL on failure, not MAP_FAILED */
			tls_last_alloc.page_type = VMEM_PAGE_REGULAR;
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
		#define PEXTRA	PROT_EXEC
	#else
		#define MEXTRA 0
		#define PEXTRA	0
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
 * 
 * This function is now NUMA-aware when vmem_setNumaNode() has been called.
 * It automatically tries 1GB pages, falls back to 2MB, then regular pages,
 * and applies madvise/mlock optimizations.
 * ============================================================================ */

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
	
	tls_last_alloc.page_type = VMEM_PAGE_2MB;  /* Windows large pages are typically 2MB */
	tls_last_alloc.numa_node = tls_numa_node;
	tls_last_alloc.is_locked = false;
	
	if (tls_numa_node >= 0) {
		/* NUMA-aware allocation on Windows */
		mem = VirtualAllocExNuma(
			GetCurrentProcess(),
			NULL,
			alignSize(bytes, pageMinimum),
			MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
			PAGE_READWRITE,
			(UCHAR)tls_numa_node
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
	tls_last_alloc.page_type = VMEM_PAGE_2MB;
	tls_last_alloc.numa_node = -1;  /* macOS doesn't have NUMA */
	tls_last_alloc.is_locked = false;
	
	mem = mmap(NULL, bytes, PROT_READ | PROT_WRITE, 
		MAP_PRIVATE | MAP_ANON, VM_FLAGS_SUPERPAGE_SIZE_2MB, 0);
	if (mem == MAP_FAILED)
		mem = NULL;

#elif defined(__FreeBSD__)
	/* FreeBSD superpage allocation */
	tls_last_alloc.page_type = VMEM_PAGE_2MB;
	tls_last_alloc.numa_node = -1;
	tls_last_alloc.is_locked = false;
	
	mem = mmap(NULL, bytes, PROT_READ | PROT_WRITE, 
		MAP_PRIVATE | MAP_ANONYMOUS | MAP_ALIGNED_SUPER, -1, 0);
	if (mem == MAP_FAILED)
		mem = NULL;

#elif defined(__linux__)
	/* Linux: Full NUMA-aware allocation with fallback chain */
	mem = allocLargePagesLinux(bytes);

#elif defined(__OpenBSD__) || defined(__NetBSD__)
	/* OpenBSD/NetBSD: No huge page support */
	tls_last_alloc.page_type = VMEM_PAGE_REGULAR;
	tls_last_alloc.numa_node = -1;
	tls_last_alloc.is_locked = false;
	mem = NULL;

#else
	/* Unknown platform */
	tls_last_alloc.page_type = VMEM_PAGE_REGULAR;
	tls_last_alloc.numa_node = -1;
	tls_last_alloc.is_locked = false;
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
	/* Check if this was allocated via numa_alloc */
	/* For simplicity, always use munmap - it works for both mmap and numa_alloc 
	 * (numa_alloc uses mmap internally) */
	munmap(ptr, bytes);
#else
	munmap(ptr, bytes);
#endif
}