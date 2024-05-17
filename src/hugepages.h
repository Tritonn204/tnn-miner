#pragma once

#ifdef __cplusplus
#include <iostream>
#include <stdlib.h>
#include <string>
#endif
#include <assert.h>

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
        printf("LookupPrivilegeValue error: %lu\n", GetLastError() ); 
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
          printf("AdjustTokenPrivileges error: %lu\n", GetLastError() ); 
          #endif
          return FALSE; 
    } 

    if (GetLastError() == ERROR_NOT_ALL_ASSIGNED)

    {
          #ifdef __cplusplus
          printf("The token does not have the specified privilege. \n");
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
    std::cout << GetLastErrorAsString() << std::endl;
    // printf("regular malloc");
    #endif
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
    // #ifdef __cplusplus
    // std::cerr << "failed to allocate hugepages... using regular malloc" << std::endl;
    // #endif
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
