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

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>

#define alignSize(pos, align) (((pos - 1) / align + 1) * align)

/* Original API - unchanged for compatibility */
void* allocMemoryPages(size_t);
void setPagesRW(void*, size_t);
void setPagesRX(void*, size_t);
void setPagesRWX(void*, size_t);
void* allocLargePagesMemory(size_t);
void freePagedMemory(void*, size_t);

/* NUMA-aware thread-local context
 * Set by NUMAOptimizer::ScopedMemoryPolicy to make allocLargePagesMemory() 
 * automatically allocate on the correct NUMA node with optimal page sizes.
 */
void vmem_setNumaNode(int node);
void vmem_clearNumaNode(void);
int vmem_getNumaNode(void);

/* Feature detection */
bool vmem_isOneGbPagesAvailable(void);
bool vmem_isHugePagesAvailable(void);

/* Allocation info for diagnostics */
typedef enum {
	VMEM_PAGE_REGULAR = 0,
	VMEM_PAGE_2MB = 1,
	VMEM_PAGE_1GB = 2
} vmem_page_type_t;

typedef struct {
	vmem_page_type_t page_type;
	int numa_node;        /* -1 if not NUMA-bound */
	bool is_locked;       /* mlock succeeded */
} vmem_alloc_info_t;

/* Get info about last allocation (thread-local) */
vmem_alloc_info_t vmem_getLastAllocInfo(void);

#ifdef __cplusplus
}
#endif