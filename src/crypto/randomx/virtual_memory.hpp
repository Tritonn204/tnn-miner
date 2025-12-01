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

#ifndef VIRTUAL_MEMORY_HPP
#define VIRTUAL_MEMORY_HPP

#include <stddef.h>
#include <stdint.h>

/* Page types for allocation info */
typedef enum {
    VMEM_PAGE_REGULAR = 0,
    VMEM_PAGE_2MB = 1,
    VMEM_PAGE_1GB = 2
} vmem_page_type_t;

/* Allocation result info */
typedef struct {
    vmem_page_type_t page_type;
    int numa_node;
    bool is_locked;
} vmem_alloc_info_t;

/* NUMA context - compatibility shims */
void vmem_setNumaNode(int node);
void vmem_clearNumaNode(void);
int vmem_getNumaNode(void);
vmem_alloc_info_t vmem_getLastAllocInfo(void);

/* Memory allocation */
void* allocMemoryPages(size_t bytes);

extern "C" {
void* allocLargePagesMemory(size_t bytes);
void freePagedMemory(void* ptr, size_t bytes);
}

/* Page protection */
void setPagesRW(void* ptr, size_t bytes);
void setPagesRX(void* ptr, size_t bytes);
void setPagesRWX(void* ptr, size_t bytes);

/* Utility */
inline constexpr size_t alignSize(size_t size, size_t align) {
    return (size + align - 1) & ~(align - 1);
}

#endif // VIRTUAL_MEMORY_HPP