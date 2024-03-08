#include <iostream>
#include <chrono>
#include <vector>
#include <windows.h>

using namespace std;
using namespace chrono;

// Computing function: Sum of elements in an array
long long sumElements(const vector<int>& arr) {
    long long sum = 0;
    for (int element : arr) {
        sum += element;
    }
    return sum;
}

#pragma once

#include <assert.h>
#include <stdlib.h>
#include <string>

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

BOOL SetPrivilege(
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
        printf("LookupPrivilegeValue error: %u\n", GetLastError() ); 
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
          printf("AdjustTokenPrivileges error: %u\n", GetLastError() ); 
          return FALSE; 
    } 

    if (GetLastError() == ERROR_NOT_ALL_ASSIGNED)

    {
          printf("The token does not have the specified privilege. \n");
          return FALSE;
    } 

    return TRUE;
}

std::string GetLastErrorAsString()
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

inline void *malloc_huge_pages(size_t size)
{
  size_t real_size = ALIGN_TO_PAGE_SIZE(size + HUGE_PAGE_SIZE);
  char *ptr = NULL;
  #if defined(_WIN32)
  ptr = (char*)VirtualAlloc(NULL, real_size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
  if (ptr == NULL) {
    std::cout << GetLastErrorAsString() << std::endl;
    ptr = (char *)malloc(real_size);
    if (ptr == NULL) return NULL;
    real_size = 0;
  }
  #else
  ptr = (char *) mmap(0, real_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | 
     MAP_HUGETLB, -1, 0);
  if (ptr == MAP_FAILED) {
    std::cerr << "failed to allocate hugepages... using regular malloc" << std::endl;
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

// Adjust the sumElements function to work directly with the allocated array
long long sumElementsArray(int* arr, size_t N) {
    long long sum = 0;
    for (size_t i = 0; i < N; ++i) {
        sum += arr[i];
    }
    return sum;
}

void performRandomAccessOperation(int* data, size_t dataSize, size_t accessCount) {
    // Seed the random number generator for reproducibility
    srand(42);

    for (size_t i = 0; i < accessCount; ++i) {
        // Generate a random index to access within the data array
        size_t index = rand() % dataSize;
        
        // Perform some operation on the data at the randomly selected index
        data[index] += 1;

        // Optionally, access another random element and perform a computation
        size_t index2 = rand() % dataSize;
        data[index] += data[index2] - data[index];
    }
}

int main() {
    HANDLE hSelfToken = NULL;

        ::OpenProcessToken(::GetCurrentProcess(), TOKEN_ALL_ACCESS, &hSelfToken);
    if (SetPrivilege(hSelfToken, SE_LOCK_MEMORY_NAME, true))
      std::cout << "Permission Granted for Huge Pages!" << std::endl;
    else
      std::cout << "Huge Pages: Permission Failed..." << std::endl;

    const size_t N = 100000000; // 100 million elements
    const size_t accessCount = 500000000; // 500 million accesses

    // Allocate memory with standard allocation
    int* standardAllocData = new int[N]{};

    // Allocate memory using huge pages
    int* hugePageData = static_cast<int*>(malloc_huge_pages(N * sizeof(int)));
    if (hugePageData == nullptr) {
        cerr << "Failed to allocate memory with huge pages." << endl;
        delete[] standardAllocData;
        return 1;
    }

    auto start = high_resolution_clock::now();
    performRandomAccessOperation(standardAllocData, N, accessCount);
    auto end = high_resolution_clock::now();
    auto duration_standard = duration_cast<milliseconds>(end - start).count();
    cout << "Operation with standard allocation: Time: " << duration_standard << " ms" << endl;

    start = high_resolution_clock::now();
    performRandomAccessOperation(hugePageData, N, accessCount);
    end = high_resolution_clock::now();
    auto duration_huge_pages = duration_cast<milliseconds>(end - start).count();
    cout << "Operation with huge pages: Time: " << duration_huge_pages << " ms" << endl;

    // Cleanup
    delete[] standardAllocData;
    free_huge_pages(hugePageData);

    return 0;
}