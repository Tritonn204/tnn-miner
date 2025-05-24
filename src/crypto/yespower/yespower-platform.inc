/*-
 * Copyright 2013-2018,2022 Alexander Peslyak
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include <errno.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#ifdef __unix__
#include <sys/mman.h>
#endif
#ifdef __linux__
#include <linux/mman.h> /* for MAP_HUGE_2MB */
#endif
#endif

#define HUGEPAGE_THRESHOLD       (12 * 1024 * 1024)

#ifdef __x86_64__
#define HUGEPAGE_SIZE           (2 * 1024 * 1024)
#else
#undef HUGEPAGE_SIZE
#endif

static void *alloc_region(yespower_region_t *region, size_t size)
{
    size_t base_size = size;
    uint8_t *base, *aligned;

#ifdef _WIN32
    // Windows implementation
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    size_t alignment = si.dwPageSize;  // Use system page size for alignment

    // Allocate extra space for alignment
    base_size = size + (alignment - 1);
    
    #if defined(__x86_64__) && defined(HUGEPAGE_SIZE)
    // Try to use large pages if available and size is suitable
    if (size >= HUGEPAGE_THRESHOLD) {
        HANDLE hToken;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken)) {
            TOKEN_PRIVILEGES tp;
            tp.PrivilegeCount = 1;
            tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
            
            if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid)) {
                AdjustTokenPrivileges(hToken, FALSE, &tp, 0, NULL, 0);
                // Note: Error handling omitted for brevity
            }
            CloseHandle(hToken);
        }
        
        base = (uint8_t*)VirtualAlloc(NULL, base_size, 
                                     MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                                     PAGE_READWRITE);
    }
    #endif
    
    // Fall back to regular allocation if large pages failed or weren't attempted
    if (!base) {
        base = (uint8_t*)VirtualAlloc(NULL, base_size, 
                                     MEM_COMMIT | MEM_RESERVE,
                                     PAGE_READWRITE);
    }
    
    if (!base) {
        // Try regular malloc as last resort
        base = (uint8_t*)malloc(base_size);
    }
    
    if (base) {
        // Align the pointer
        aligned = (uint8_t*)(((uintptr_t)base + (alignment - 1)) & ~(alignment - 1));
    } else {
        base = aligned = NULL;
        base_size = 0;
    }

#else
    // Original Unix implementation
#ifdef MAP_ANON
    int flags =
#ifdef MAP_NOCORE
        MAP_NOCORE |
#endif
        MAP_ANON | MAP_PRIVATE;
#if defined(MAP_HUGETLB) && defined(MAP_HUGE_2MB) && defined(HUGEPAGE_SIZE)
    size_t new_size = size;
    const size_t hugepage_mask = (size_t)HUGEPAGE_SIZE - 1;
    if (size >= HUGEPAGE_THRESHOLD && size + hugepage_mask >= size) {
        flags |= MAP_HUGETLB | MAP_HUGE_2MB;
        new_size = size + hugepage_mask;
        new_size &= ~hugepage_mask;
    }
    base = mmap(NULL, new_size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (base != MAP_FAILED) {
        base_size = new_size;
    } else if (flags & MAP_HUGETLB) {
        flags &= ~(MAP_HUGETLB | MAP_HUGE_2MB);
        base = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    }
#else
    base = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, -1, 0);
#endif
    if (base == MAP_FAILED)
        base = NULL;
    aligned = base;
#elif defined(HAVE_POSIX_MEMALIGN)
    if ((errno = posix_memalign((void **)&base, 64, size)) != 0)
        base = NULL;
    aligned = base;
#else
    base = aligned = NULL;
    if (size + 63 < size) {
        errno = ENOMEM;
    } else if ((base = malloc(size + 63)) != NULL) {
        aligned = base + 63;
        aligned -= (uintptr_t)aligned & 63;
    }
#endif
#endif // _WIN32

    region->base = base;
    region->aligned = aligned;
    region->base_size = base ? base_size : 0;
    region->aligned_size = base ? size : 0;
    return aligned;
}

static inline void init_region(yespower_region_t *region)
{
    region->base = region->aligned = NULL;
    region->base_size = region->aligned_size = 0;
}

static int free_region(yespower_region_t *region)
{
    if (region->base) {
#ifdef _WIN32
        // Windows implementation
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery(region->base, &mbi, sizeof(mbi)) == sizeof(mbi)) {
            if (mbi.State == MEM_COMMIT) {
                // If it was allocated with VirtualAlloc
                if (!VirtualFree(region->base, 0, MEM_RELEASE)) {
                    // If VirtualFree fails, try regular free
                    free(region->base);
                }
            } else {
                // If it was allocated with malloc
                free(region->base);
            }
        }
#else
#ifdef MAP_ANON
        if (munmap(region->base, region->base_size))
            return -1;
#else
        free(region->base);
#endif
#endif
    }
    init_region(region);
    return 0;
}