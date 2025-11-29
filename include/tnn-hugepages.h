#pragma once

#ifdef __cplusplus
#include <iostream>
#include <cstdlib>
#include <string>
#endif

#include <assert.h>

#include "terminal.h"

extern bool printHugepagesError;

#if defined(_WIN32)
  #include <Windows.h>
  #pragma comment(lib, "advapi32.lib")
#else
  #include <sys/mman.h>
  #include <errno.h>
  #include <string.h>
  #include <unistd.h>
#endif

#define HUGE_META_PAGE_SIZE (2ULL * 1024 * 1024)
#define HUGE_PAGE_2MB       (2ULL * 1024 * 1024)
#define HUGE_PAGE_1GB       (1024ULL * 1024 * 1024)

#define ALIGN_UP(x, a)  ( ((x) + (a) - 1) & ~((size_t)((a) - 1)) )

#ifdef _WIN32

#ifdef __cplusplus
inline BOOL SetPrivilege(
    HANDLE hToken,          // access token handle
    LPCTSTR lpszPrivilege,  // name of privilege to enable/disable
    BOOL bEnablePrivilege   // to enable or disable privilege
    )
{
    TOKEN_PRIVILEGES tp;
    LUID luid;

    if (!LookupPrivilegeValue(
            NULL,
            lpszPrivilege,
            &luid))
    {
        setcolor(RED);
        printf("LookupPrivilegeValue error: %lu\n", GetLastError());
        fflush(stdout);
        return FALSE;
    }

    tp.PrivilegeCount = 1;
    tp.Privileges[0].Luid = luid;
    tp.Privileges[0].Attributes = bEnablePrivilege ? SE_PRIVILEGE_ENABLED : 0;

    if (!AdjustTokenPrivileges(
            hToken,
            FALSE,
            &tp,
            sizeof(TOKEN_PRIVILEGES),
            (PTOKEN_PRIVILEGES)NULL,
            (PDWORD)NULL))
    {
        setcolor(RED);
        printf("AdjustTokenPrivileges error: %lu\n", GetLastError());
        fflush(stdout);
        return FALSE;
    }

    if (GetLastError() == ERROR_NOT_ALL_ASSIGNED)
    {
        setcolor(RED);
        printf("The token does not have the specified privilege.\n");
        fflush(stdout);
        return FALSE;
    }

    return TRUE;
}

inline std::string GetLastErrorAsString()
{
    DWORD errorMessageID = ::GetLastError();
    if (errorMessageID == 0) {
        return std::string();
    }

    LPSTR messageBuffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM     |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        errorMessageID,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&messageBuffer,
        0,
        NULL
    );

    std::string message(messageBuffer, size);
    LocalFree(messageBuffer);
    return message;
}
#endif // __cplusplus

#endif // _WIN32

// ---- Linux 1GB hugepage macros (if not already defined) ----
#if !defined(_WIN32) && !defined(__APPLE__)
  #ifndef MAP_HUGE_SHIFT
    #define MAP_HUGE_SHIFT 26
  #endif
  #ifndef MAP_HUGE_2MB
    #define MAP_HUGE_2MB   (21 << MAP_HUGE_SHIFT)
  #endif
  #ifndef MAP_HUGE_1GB
    #define MAP_HUGE_1GB   (30 << MAP_HUGE_SHIFT)
  #endif
#endif

inline void* malloc_huge_pages(size_t size)
{
    size_t requested = size + HUGE_META_PAGE_SIZE;
    char*  ptr       = nullptr;
    size_t real_size = 0;

#if defined(_WIN32)

    SIZE_T large_page_size = GetLargePageMinimum();
    if (large_page_size != 0) {
        HANDLE hToken;
        if (OpenProcessToken(GetCurrentProcess(),
                             TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                             &hToken))
        {
            SetPrivilege(hToken, TEXT("SeLockMemoryPrivilege"), TRUE);
            CloseHandle(hToken);
        }

        real_size = ALIGN_UP(requested, large_page_size);

        ptr = (char*)VirtualAlloc(
            NULL,
            real_size,
            MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
            PAGE_READWRITE
        );
    }

    if (ptr == NULL) {
        // Large pages failed; fallback to normal VirtualAlloc or malloc
        if (printHugepagesError) {
#ifdef __cplusplus
            std::cerr << GetLastErrorAsString() << std::endl;
#endif
            printHugepagesError = false;
        }

        SYSTEM_INFO si;
        GetSystemInfo(&si);
        SIZE_T page_size = si.dwPageSize ? si.dwPageSize : 4096;

        real_size = ALIGN_UP(requested, page_size);
        ptr = (char*)VirtualAlloc(
            NULL,
            real_size,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE
        );

        if (ptr == NULL) {
            // Last resort: malloc
            ptr = (char*)std::malloc(real_size);
            if (ptr == NULL) return NULL;
            real_size = 0; // mark as malloc
        }
    }

#else // POSIX path (Linux / others)

    // On Linux, try 1GB huge pages first for very large allocations, then 2MB.
    int use_huge = 0;
    int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;

#if !defined(__APPLE__)
    // Attempt 1GB huge pages if request is large enough
    if (requested >= HUGE_PAGE_1GB) {
        size_t huge_gran = HUGE_PAGE_1GB;
        real_size = ALIGN_UP(requested, huge_gran);
        mmap_flags |= MAP_HUGETLB | MAP_HUGE_1GB;

        ptr = (char*)mmap(
            0,
            real_size,
            PROT_READ | PROT_WRITE,
            mmap_flags,
            -1,
            0
        );

        if (ptr != MAP_FAILED) {
            use_huge = 1;
        } else {
            ptr = nullptr;
            real_size = 0;
            mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS; // reset
        }
    }

    if (!ptr) {
        size_t huge_gran = HUGE_PAGE_2MB;
        real_size = ALIGN_UP(requested, huge_gran);
        mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB;

        ptr = (char*)mmap(
            0,
            real_size,
            PROT_READ | PROT_WRITE,
            mmap_flags,
            -1,
            0
        );

        if (ptr != MAP_FAILED) {
            use_huge = 1;
        } else {
            ptr = nullptr;
            real_size = 0;
        }
    }
#endif // !__APPLE__

    if (!ptr) {
        if (printHugepagesError) {
#ifdef __cplusplus
            std::cerr << "failed to allocate hugepages... using regular malloc"
                      << std::endl;
#endif
            printHugepagesError = false;
        }

        real_size = ALIGN_UP(requested, HUGE_META_PAGE_SIZE);
        ptr = (char*)std::malloc(real_size);
        if (ptr == NULL) return NULL;
        real_size = 0;
    }

#endif // _WIN32 / POSIX

    *((size_t*)ptr) = real_size;

    return ptr + HUGE_META_PAGE_SIZE;
}

inline void free_huge_pages(void* ptr)
{
    if (ptr == NULL) return;

    void* real_ptr = (char*)ptr - HUGE_META_PAGE_SIZE;

    size_t real_size = *((size_t*)real_ptr);

    if (real_size != 0) {
#if defined(_WIN32)
        VirtualFree(real_ptr, 0, MEM_RELEASE);
#else
        munmap(real_ptr, real_size);
#endif
    } else {
        std::free(real_ptr);
    }
}
