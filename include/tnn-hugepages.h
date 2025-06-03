#pragma once

#ifdef __cplusplus
#include <iostream>
#include <stdlib.h>
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
#endif

#define HUGE_PAGE_SIZE (2 * 1024*1024)
#define ALIGN_TO_PAGE_SIZE(x) \
  (((x) + HUGE_PAGE_SIZE - 1) / HUGE_PAGE_SIZE * HUGE_PAGE_SIZE)

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

    if ( !LookupPrivilegeValue( 
            NULL,            // lookup privilege on local system
            lpszPrivilege,   // privilege to lookup 
            &luid ) )        // receives LUID of privilege
    {
        #ifdef __cplusplus
        setcolor(RED);
        printf("LookupPrivilegeValue error: %lu\n", GetLastError() ); 
        fflush(stdout);
        #endif
        return FALSE; 
    }

    tp.PrivilegeCount = 1;
    tp.Privileges[0].Luid = luid;
    if (bEnablePrivilege)
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    else
        tp.Privileges[0].Attributes = 0;

    // Enable the privilege or disable all privileges.

    if ( !AdjustTokenPrivileges(
           hToken, 
           FALSE, 
           &tp, 
           sizeof(TOKEN_PRIVILEGES), 
           (PTOKEN_PRIVILEGES) NULL, 
           (PDWORD) NULL) )
    { 
          #ifdef __cplusplus
          setcolor(RED);
          printf("AdjustTokenPrivileges error: %lu\n", GetLastError() ); 
          fflush(stdout);
          #endif
          return FALSE; 
    } 

    if (GetLastError() == ERROR_NOT_ALL_ASSIGNED)

    {
          #ifdef __cplusplus
          setcolor(RED);
          printf("The token does not have the specified privilege. \n");
          fflush(stdout);
          #endif
          return FALSE;
    } 

    return TRUE;
}

inline std::string GetLastErrorAsString()
{
    //Get the error message ID, if any.
    DWORD errorMessageID = ::GetLastError();
    if(errorMessageID == 0) {
        return std::string(); //No error message has been recorded
    }
    
    LPSTR messageBuffer = nullptr;

    //Ask Win32 to give us the string version of that message ID.
    //The parameters we pass in, tell Win32 to create the buffer that holds the message for us (because we don't yet know how long the message string will be).
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);
    
    //Copy the error message into a std::string.
    std::string message(messageBuffer, size);
    
    //Free the Win32's string's buffer.
    LocalFree(messageBuffer);
            
    return message;
}

#endif

#endif

inline void *malloc_huge_pages(size_t size)
{
  size_t real_size = ALIGN_TO_PAGE_SIZE(size + HUGE_PAGE_SIZE);
  char *ptr = NULL;
  #if defined(_WIN32)
  ptr = (char*)VirtualAlloc(NULL, real_size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
  if (ptr == NULL) {
    #ifdef __cplusplus
      if (printHugepagesError) {
        std::cerr << GetLastErrorAsString() << std::endl;
      }
    #endif
    printHugepagesError = false;
    ptr = (char *)malloc(real_size);
    if (ptr == NULL) return NULL;
    real_size = 0;
  }
  #else
  ptr = (char *) mmap(0, real_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS
    #if !defined(__APPLE__)
     | MAP_HUGETLB
    #endif
     , -1, 0);
  if (ptr == MAP_FAILED) {
    #ifdef __cplusplus
      if (printHugepagesError) {
        std::cerr << "failed to allocate hugepages... using regular malloc" << std::endl;
      }
    #endif
    printHugepagesError = false;
    ptr = (char *)malloc(real_size);
    if (ptr == NULL) return NULL;
    real_size = 0;
  }
  #endif

  *((size_t *) ptr) = real_size;

  return ptr + HUGE_PAGE_SIZE;
}

inline void free_huge_pages(void *ptr)
{
  if (ptr == NULL) return;

  void *real_ptr = (char *)ptr - HUGE_PAGE_SIZE;

  size_t real_size = *((size_t *)real_ptr);

  assert(real_size % HUGE_PAGE_SIZE == 0);

  if (real_size != 0) {
    #if defined (_WIN32)
    VirtualFree(ptr, real_size, MEM_DECOMMIT);
    #else
    munmap(real_ptr, real_size);
    #endif
  } else free(real_ptr);
}

inline bool setupHugePages() {
#ifdef __linux__
    // Check current huge pages configuration
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    size_t huge_page_size = 0;
    size_t total_huge_pages = 0;
    size_t free_huge_pages = 0;
    
    while (std::getline(meminfo, line)) {
        if (line.find("Hugepagesize:") != std::string::npos) {
            sscanf(line.c_str(), "Hugepagesize: %zu kB", &huge_page_size);
            huge_page_size *= 1024; // Convert to bytes
        } else if (line.find("HugePages_Total:") != std::string::npos) {
            sscanf(line.c_str(), "HugePages_Total: %zu", &total_huge_pages);
        } else if (line.find("HugePages_Free:") != std::string::npos) {
            sscanf(line.c_str(), "HugePages_Free: %zu", &free_huge_pages);
        }
    }
    meminfo.close();
    
    // Calculate required huge pages for RandomX
    // RandomX needs ~2.5GB per NUMA node + caches
    size_t required_memory = 0;
    if (rx_numa_enabled) {
        required_memory = (2560ULL * 1024 * 1024) * numa_nodes; // 2.5GB per node
    } else {
        required_memory = 2560ULL * 1024 * 1024; // 2.5GB total
    }
    required_memory += 512ULL * 1024 * 1024; // Add 512MB for caches and overhead
    
    size_t required_pages = (required_memory + huge_page_size - 1) / huge_page_size;
    
    setcolor(BRIGHT_YELLOW);
    std::cout << " Huge page size: " << (huge_page_size / (1024 * 1024)) << " MB" << std::endl;
    std::cout << " Total huge pages: " << total_huge_pages << std::endl;
    std::cout << " Free huge pages: " << free_huge_pages << std::endl;
    std::cout << " Required huge pages: " << required_pages << std::endl;
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    
    if (free_huge_pages < required_pages) {
        setcolor(RED);
        std::cout << "\nInsufficient huge pages available!" << std::endl;
        std::cout << "To fix this, run as root:" << std::endl;
        std::cout << "  echo " << (total_huge_pages - free_huge_pages + required_pages) 
                  << " > /proc/sys/vm/nr_hugepages" << std::endl;
        std::cout << "\nOr to set permanently, add to /etc/sysctl.conf:" << std::endl;
        std::cout << "  vm.nr_hugepages = " << (total_huge_pages - free_huge_pages + required_pages) 
                  << std::endl;
        std::cout << "\nContinuing without huge pages..." << std::endl;
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        return false;
    }
    
    return true;
    
#elif defined(_WIN32)
    // Check if we have SeLockMemoryPrivilege
    HANDLE hToken = NULL;
    BOOL hasPrivilege = FALSE;
    
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken)) {
        LUID luid;
        if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &luid)) {
            PRIVILEGE_SET privSet = {0};
            privSet.PrivilegeCount = 1;
            privSet.Control = PRIVILEGE_SET_ALL_NECESSARY;
            privSet.Privilege[0].Luid = luid;
            privSet.Privilege[0].Attributes = SE_PRIVILEGE_ENABLED;
            
            BOOL result;
            PrivilegeCheck(hToken, &privSet, &result);
            hasPrivilege = result;
        }
        CloseHandle(hToken);
    }
    
    if (!hasPrivilege) {
        setcolor(RED);
        std::cout << "\nHuge pages not available - missing SeLockMemoryPrivilege!" << std::endl;
        std::cout << "To enable:" << std::endl;
        std::cout << "1. Run gpedit.msc as Administrator" << std::endl;
        std::cout << "2. Navigate to: Computer Configuration → Windows Settings → " << std::endl;
        std::cout << "   Security Settings → Local Policies → User Rights Assignment" << std::endl;
        std::cout << "3. Add your user to 'Lock pages in memory'" << std::endl;
        std::cout << "4. Reboot the system" << std::endl;
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        return false;
    }
    
    // Check minimum large page size
    SIZE_T minLargePageSize = GetLargePageMinimum();
    if (minLargePageSize == 0) {
        setcolor(RED);
        std::cout << "\nHuge pages not supported on this Windows version!" << std::endl;
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        return false;
    }
    
    setcolor(BRIGHT_YELLOW);
    std::cout << " Huge page size: " << (minLargePageSize / (1024 * 1024)) << " MB" << std::endl;
    std::cout << " Huge pages available\n" << std::endl;
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    
    return true;
#else
    return false;
#endif
}
