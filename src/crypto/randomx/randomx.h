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

#ifndef RANDOMX_H
#define RANDOMX_H

#include <stddef.h>
#include <stdint.h>

#define RANDOMX_HASH_SIZE 32
#define RANDOMX_DATASET_ITEM_SIZE 64

#ifndef RANDOMX_EXPORT
#define RANDOMX_EXPORT
#endif

#define RANDOMX_BLOB_SIZE 76
#define RANDOMX_TEMPLATE_SIZE 352

typedef enum
{
  RANDOMX_FLAG_DEFAULT = 0,
  RANDOMX_FLAG_LARGE_PAGES = 1,
  RANDOMX_FLAG_HARD_AES = 2,
  RANDOMX_FLAG_FULL_MEM = 4,
  RANDOMX_FLAG_JIT = 8,
  RANDOMX_FLAG_SECURE = 16,
  RANDOMX_FLAG_ARGON2_SSSE3 = 32,
  RANDOMX_FLAG_ARGON2_AVX2 = 64,
  RANDOMX_FLAG_ARGON2_AVX512 = 128,
  RANDOMX_FLAG_ARGON2 = 224,
  RANDOMX_FLAG_AMD = 256
} randomx_flags;

typedef struct randomx_dataset randomx_dataset;
typedef struct randomx_cache randomx_cache;
typedef struct randomx_vm randomx_vm;

extern randomx_flags rxFlags;

extern randomx_dataset *rxDataset;
extern randomx_cache *rxCache;
extern randomx_cache *rxCache_dev;
extern randomx_dataset *rxDatasets_numa[256];

extern bool rx_numa_enabled;

#if defined(__cplusplus)

#ifdef __cpp_constexpr
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

inline CONSTEXPR randomx_flags operator|(randomx_flags a, randomx_flags b)
{
  return static_cast<randomx_flags>(static_cast<int>(a) | static_cast<int>(b));
}
inline CONSTEXPR randomx_flags operator&(randomx_flags a, randomx_flags b)
{
  return static_cast<randomx_flags>(static_cast<int>(a) & static_cast<int>(b));
}
inline randomx_flags &operator|=(randomx_flags &a, randomx_flags b)
{
  return a = a | b;
}

extern "C"
{
#endif

  /**
   * @return The recommended flags to be used on the current machine.
   *         Does not include:
   *            RANDOMX_FLAG_LARGE_PAGES
   *            RANDOMX_FLAG_FULL_MEM
   *            RANDOMX_FLAG_SECURE
   *         These flags must be added manually if desired.
   *         On OpenBSD RANDOMX_FLAG_SECURE is enabled by default in JIT mode as W^X is enforced by the OS.
   */
  RANDOMX_EXPORT randomx_flags randomx_get_flags(void);

  /**
   * Creates a randomx_cache structure and allocates memory for RandomX Cache.
   *
   * @param flags is any combination of these 2 flags (each flag can be set or not set):
   *        RANDOMX_FLAG_LARGE_PAGES - allocate memory in large pages
   *        RANDOMX_FLAG_JIT - create cache structure with JIT compilation support; this makes
   *                           subsequent Dataset initialization faster
   *        Optionally, one of these two flags may be selected:
   *        RANDOMX_FLAG_ARGON2_SSSE3 - optimized Argon2 for CPUs with the SSSE3 instruction set
   *                                   makes subsequent cache initialization faster
   *        RANDOMX_FLAG_ARGON2_AVX2 - optimized Argon2 for CPUs with the AVX2 instruction set
   *                                   makes subsequent cache initialization faster
   *        RANDOMX_FLAG_ARGON2_AVX512 - optimized Argon2 for CPUs with the AVX512 instruction set
   *                                   makes subsequent cache initialization faster
   * @return Pointer to an allocated randomx_cache structure.
   *         Returns NULL if:
   *         (1) memory allocation fails
   *         (2) the RANDOMX_FLAG_JIT is set and JIT compilation is not supported on the current platform
   *         (3) an invalid or unsupported RANDOMX_FLAG_ARGON2 value is set
   */
  RANDOMX_EXPORT randomx_cache *randomx_alloc_cache(randomx_flags flags);

  /**
   * Initializes the cache memory and SuperscalarHash using the provided key value.
   * Does nothing if called again with the same key value.
   *
   * @param cache is a pointer to a previously allocated randomx_cache structure. Must not be NULL.
   * @param key is a pointer to memory which contains the key value. Must not be NULL.
   * @param keySize is the number of bytes of the key.
   */
  RANDOMX_EXPORT void randomx_init_cache(randomx_cache *cache, const void *key, size_t keySize);

  /**
   * Releases all memory occupied by the randomx_cache structure.
   *
   * @param cache is a pointer to a previously allocated randomx_cache structure.
   */
  RANDOMX_EXPORT void randomx_release_cache(randomx_cache *cache);

  /**
   * Creates a randomx_dataset structure and allocates memory for RandomX Dataset.
   *
   * @param flags is the initialization flags. Only one flag is supported (can be set or not set):
   *        RANDOMX_FLAG_LARGE_PAGES - allocate memory in large pages
   *
   * @return Pointer to an allocated randomx_dataset structure.
   *         NULL is returned if memory allocation fails.
   */
  RANDOMX_EXPORT randomx_dataset *randomx_alloc_dataset(randomx_flags flags);

  /**
   * Gets the number of items contained in the dataset.
   *
   * @return the number of items contained in the dataset.
   */
  RANDOMX_EXPORT unsigned long randomx_dataset_item_count(void);

  /**
   * Initializes dataset items.
   *
   * Note: In order to use the Dataset, all items from 0 to (randomx_dataset_item_count() - 1) must be initialized.
   * This may be done by several calls to this function using non-overlapping item sequences.
   *
   * @param dataset is a pointer to a previously allocated randomx_dataset structure. Must not be NULL.
   * @param cache is a pointer to a previously allocated and initialized randomx_cache structure. Must not be NULL.
   * @param startItem is the item number where intialization should start.
   * @param itemCount is the number of items that should be initialized.
   */
  RANDOMX_EXPORT void randomx_init_dataset(randomx_dataset *dataset, randomx_cache *cache, unsigned long startItem, unsigned long itemCount);

  /**
   * Returns a pointer to the internal memory buffer of the dataset structure. The size
   * of the internal memory buffer is randomx_dataset_item_count() * RANDOMX_DATASET_ITEM_SIZE.
   *
   * @param dataset is a pointer to a previously allocated randomx_dataset structure. Must not be NULL.
   *
   * @return Pointer to the internal memory buffer of the dataset structure.
   */
  RANDOMX_EXPORT void *randomx_get_dataset_memory(randomx_dataset *dataset);

  /**
   * Releases all memory occupied by the randomx_dataset structure.
   *
   * @param dataset is a pointer to a previously allocated randomx_dataset structure.
   */
  RANDOMX_EXPORT void randomx_release_dataset(randomx_dataset *dataset);

  /**
   * Creates and initializes a RandomX virtual machine.
   *
   * @param flags is any combination of these 5 flags (each flag can be set or not set):
   *        RANDOMX_FLAG_LARGE_PAGES - allocate scratchpad memory in large pages
   *        RANDOMX_FLAG_HARD_AES - virtual machine will use hardware accelerated AES
   *        RANDOMX_FLAG_FULL_MEM - virtual machine will use the full dataset
   *        RANDOMX_FLAG_JIT - virtual machine will use a JIT compiler
   *        RANDOMX_FLAG_SECURE - when combined with RANDOMX_FLAG_JIT, the JIT pages are never
   *                              writable and executable at the same time (W^X policy)
   *        The numeric values of the first 4 flags are ordered so that a higher value will provide
   *        faster hash calculation and a lower numeric value will provide higher portability.
   *        Using RANDOMX_FLAG_DEFAULT (all flags not set) works on all platforms, but is the slowest.
   * @param cache is a pointer to an initialized randomx_cache structure. Can be
   *        NULL if RANDOMX_FLAG_FULL_MEM is set.
   * @param dataset is a pointer to a randomx_dataset structure. Can be NULL
   *        if RANDOMX_FLAG_FULL_MEM is not set.
   *
   * @return Pointer to an initialized randomx_vm structure.
   *         Returns NULL if:
   *         (1) Scratchpad memory allocation fails.
   *         (2) The requested initialization flags are not supported on the current platform.
   *         (3) cache parameter is NULL and RANDOMX_FLAG_FULL_MEM is not set
   *         (4) dataset parameter is NULL and RANDOMX_FLAG_FULL_MEM is set
   */
  RANDOMX_EXPORT randomx_vm *randomx_create_vm(randomx_flags flags, randomx_cache *cache, randomx_dataset *dataset);

  /**
   * Reinitializes a virtual machine with a new Cache. This function should be called anytime
   * the Cache is reinitialized with a new key. Does nothing if called with a Cache containing
   * the same key value as already set.
   *
   * @param machine is a pointer to a randomx_vm structure that was initialized
   *        without RANDOMX_FLAG_FULL_MEM. Must not be NULL.
   * @param cache is a pointer to an initialized randomx_cache structure. Must not be NULL.
   */
  RANDOMX_EXPORT void randomx_vm_set_cache(randomx_vm *machine, randomx_cache *cache);

  /**
   * Reinitializes a virtual machine with a new Dataset.
   *
   * @param machine is a pointer to a randomx_vm structure that was initialized
   *        with RANDOMX_FLAG_FULL_MEM. Must not be NULL.
   * @param dataset is a pointer to an initialized randomx_dataset structure. Must not be NULL.
   */
  RANDOMX_EXPORT void randomx_vm_set_dataset(randomx_vm *machine, randomx_dataset *dataset);

  /**
   * Releases all memory occupied by the randomx_vm structure.
   *
   * @param machine is a pointer to a previously created randomx_vm structure.
   */
  RANDOMX_EXPORT void randomx_destroy_vm(randomx_vm *machine);

  /**
   * Calculates a RandomX hash value.
   *
   * @param machine is a pointer to a randomx_vm structure. Must not be NULL.
   * @param input is a pointer to memory to be hashed. Must not be NULL.
   * @param inputSize is the number of bytes to be hashed.
   * @param output is a pointer to memory where the hash will be stored. Must not
   *        be NULL and at least RANDOMX_HASH_SIZE bytes must be available for writing.
   */
  RANDOMX_EXPORT void randomx_calculate_hash(randomx_vm *machine, const void *input, size_t inputSize, void *output);

  /**
   * Set of functions used to calculate multiple RandomX hashes more efficiently.
   * randomx_calculate_hash_first will begin a hash calculation.
   * randomx_calculate_hash_next  will output the hash value of the previous input
   *                              and begin the calculation of the next hash.
   * randomx_calculate_hash_last  will output the hash value of the previous input.
   *
   * WARNING: These functions may alter the floating point rounding mode of the calling thread.
   *
   * @param machine is a pointer to a randomx_vm structure. Must not be NULL.
   * @param input is a pointer to memory to be hashed. Must not be NULL.
   * @param inputSize is the number of bytes to be hashed.
   * @param nextInput is a pointer to memory to be hashed for the next hash. Must not be NULL.
   * @param nextInputSize is the number of bytes to be hashed for the next hash.
   * @param output is a pointer to memory where the hash will be stored. Must not
   *        be NULL and at least RANDOMX_HASH_SIZE bytes must be available for writing.
   */
  RANDOMX_EXPORT void randomx_calculate_hash_first(randomx_vm *machine, const void *input, size_t inputSize);
  RANDOMX_EXPORT void randomx_calculate_hash_next(randomx_vm *machine, const void *nextInput, size_t nextInputSize, void *output);
  RANDOMX_EXPORT void randomx_calculate_hash_last(randomx_vm *machine, void *output);

  /**
   * Calculate a RandomX commitment from a RandomX hash and its input.
   *
   * @param input is a pointer to memory that was hashed. Must not be NULL.
   * @param inputSize is the number of bytes in the input.
   * @param hash_in is the output from randomx_calculate_hash* (RANDOMX_HASH_SIZE bytes).
   * @param com_out is a pointer to memory where the commitment will be stored. Must not
   *        be NULL and at least RANDOMX_HASH_SIZE bytes must be available for writing.
   */
  RANDOMX_EXPORT void randomx_calculate_commitment(const void *input, size_t inputSize, const void *hash_in, void *com_out);

// Why in the hell did I commit the sorcery seen below instead of just putting it in randomx.cpp? REVIEWME later
#if defined(__cplusplus)
}

#ifdef _WIN32
#include <windows.h>
#endif

#include <fstream>
#include "numa_optimizer.hpp"
#include "terminal.hpp"

inline bool setupHugePagesRX()
{
#ifdef __linux__
  // Check current huge pages configuration
  HugePagesInfo hpinfo = getHugePagesInfo();

  int numa_nodes = NUMAOptimizer::getMemoryNodes();

  // Calculate required memory for RandomX
  size_t required_memory = 0;
  if (rx_numa_enabled)
  {
    required_memory = (2560ULL * 1024 * 1024) * numa_nodes;
  }
  else
  {
    required_memory = 2560ULL * 1024 * 1024;
  }
  required_memory += 512ULL * 1024 * 1024;

  // Calculate available memory from huge pages (combining both sizes)
  size_t available_hp_memory = (hpinfo.free_2mb * hpinfo.page_size_2mb) +
                               (hpinfo.free_1gb * hpinfo.page_size_1gb);

  setcolor(BRIGHT_YELLOW);
  if (hpinfo.page_size_1gb > 0 && hpinfo.total_1gb > 0)
  {
    std::cout << " 1GB huge pages: " << hpinfo.free_1gb << " free / " << hpinfo.total_1gb << " total" << std::endl;
  }
  if (hpinfo.page_size_2mb > 0)
  {
    std::cout << " 2MB huge pages: " << hpinfo.free_2mb << " free / " << hpinfo.total_2mb << " total" << std::endl;
  }
  std::cout << " Required memory: " << (required_memory / (1024 * 1024)) << " MB" << std::endl;
  std::cout << " Available huge page memory: " << (available_hp_memory / (1024 * 1024)) << " MB" << std::endl;
  fflush(stdout);
  setcolor(BRIGHT_WHITE);

  if (available_hp_memory >= required_memory)
  {
    return true;
  }

  // Insufficient - try auto-allocate 2MB pages if root
  size_t shortfall = required_memory - available_hp_memory;
  size_t additional_2mb_pages = (shortfall + hpinfo.page_size_2mb - 1) / hpinfo.page_size_2mb;
  size_t needed_total = hpinfo.total_2mb + additional_2mb_pages;

  if (geteuid() == 0)
  {
    setcolor(BRIGHT_YELLOW);
    std::cout << " Attempting to allocate " << needed_total << " 2MB huge pages..." << std::endl;
    fflush(stdout);

    std::ofstream nr_hugepages("/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages");
    if (nr_hugepages.is_open())
    {
      nr_hugepages << needed_total;
      nr_hugepages.close();

      // Verify
      std::ifstream verify("/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages");
      size_t new_free = 0;
      if (verify.is_open())
      {
        verify >> new_free;
        verify.close();
      }

      size_t new_available = (new_free * hpinfo.page_size_2mb) +
                             (hpinfo.free_1gb * hpinfo.page_size_1gb);

      if (new_available >= required_memory)
      {
        setcolor(GREEN);
        std::cout << " Successfully allocated huge pages" << std::endl;
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        return true;
      }
      else
      {
        setcolor(RED);
        std::cout << " Allocation partially succeeded - system may have insufficient free memory" << std::endl;
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
      }
    }
  }

  // Failed - print instructions
  setcolor(RED);
  std::cout << "\nInsufficient huge pages available!" << std::endl;
  if (geteuid() != 0)
  {
    std::cout << "Run as root for automatic huge page allocation, or manually run:" << std::endl;
  }
  else
  {
    std::cout << "To fix this manually, run:" << std::endl;
  }
  std::cout << "  echo " << needed_total << " > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages" << std::endl;
  std::cout << "\nOr to set permanently, add to /etc/sysctl.conf:" << std::endl;
  std::cout << "  vm.nr_hugepages = " << needed_total << std::endl;
  std::cout << "\nContinuing without huge pages..." << std::endl;
  fflush(stdout);
  setcolor(BRIGHT_WHITE);
  return false;

#elif defined(_WIN32)
  // Check if we have SeLockMemoryPrivilege
  HANDLE hToken = NULL;
  BOOL hasPrivilege = FALSE;

  if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken))
  {
    LUID luid;
    if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &luid))
    {
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

  if (!hasPrivilege)
  {
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
  if (minLargePageSize == 0)
  {
    setcolor(RED);
    std::cout << "\nHuge pages not supported on this Windows version!" << std::endl;
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
    return false;
  }

  setcolor(BRIGHT_YELLOW);
  std::cout << " Huge page size: " << (minLargePageSize / (1024 * 1024)) << " MB" << std::endl;
  std::cout << " Huge pages available\n"
            << std::endl;
  fflush(stdout);
  setcolor(BRIGHT_WHITE);

  return true;
#else
  return false;
#endif
}

#endif

#endif