/*
 * divsufsort_private.h for libdivsufsort
 * Copyright (c) 2003-2008 Yuta Mori All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef _DIVSUFSORT_PRIVATE_CUDA_H
#define _DIVSUFSORT_PRIVATE_CUDA_H 1

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#if HAVE_CONFIG_H
# include "config.h"
#endif
#include <assert.h>
#include <stdio.h>
#if HAVE_STRING_H
# include <string.h>
#endif
#if HAVE_STDLIB_H
# include <stdlib.h>
#endif
#if HAVE_MEMORY_H
# include <memory.h>
#endif
#if HAVE_STDDEF_H
# include <stddef.h>
#endif
#if HAVE_STRINGS_H
# include <strings.h>
#endif
#if HAVE_INTTYPES_H
# include <inttypes.h>
#else
# if HAVE_STDINT_H
#  include <stdint.h>
# endif
#endif
#if defined(BUILD_DIVSUFSORT64)
# ifndef SAIDX_T
#  define SAIDX_T
#  define saidx_t saidx64_t
# endif /* SAIDX_T */
# ifndef PRIdSAIDX_T
#  define PRIdSAIDX_T PRIdSAIDX64_T
# endif /* PRIdSAIDX_T */
# define divsufsort divsufsort64
# define divbwt divbwt64
# define divsufsort_version divsufsort64_version
# define bw_transform bw_transform64
# define inverse_bw_transform inverse_bw_transform64
# define sufcheck sufcheck64
# define sa_search sa_search64
# define sa_simplesearch sa_simplesearch64
# define sssort sssort64
# define trsort trsort64
#else
# include "divsufsort_cuda.cuh"
#endif


/*- Constants -*/
#if !defined(UINT8_MAX_CUDA_CUDA)
# define UINT8_MAX_CUDA_CUDA (255)
#endif /* UINT8_MAX_CUDA */
#if defined(ALPHABET_SIZE_CUDA) && (ALPHABET_SIZE_CUDA < 1)
# undef ALPHABET_SIZE_CUDA
#endif
#if !defined(ALPHABET_SIZE_CUDA)
# define ALPHABET_SIZE_CUDA (UINT8_MAX_CUDA_CUDA + 1)
#endif
/* for divsufsort.c */
#define BUCKET_A_SIZE_CUDA (ALPHABET_SIZE_CUDA)
#define BUCKET_B_SIZE_CUDA (ALPHABET_SIZE_CUDA * ALPHABET_SIZE_CUDA)
/* for sssort.c */
#if defined(SS_INSERTIONSORT_THRESHOLD_CUDA)
# if SS_INSERTIONSORT_THRESHOLD_CUDA < 1
#  undef SS_INSERTIONSORT_THRESHOLD_CUDA
#  define SS_INSERTIONSORT_THRESHOLD_CUDA (1)
# endif
#else
# define SS_INSERTIONSORT_THRESHOLD_CUDA (8)
#endif
#if defined(SS_BLOCKSIZE_CUDA)
# if SS_BLOCKSIZE_CUDA < 0
#  undef SS_BLOCKSIZE_CUDA
#  define SS_BLOCKSIZE_CUDA (0)
# elif 32768 <= SS_BLOCKSIZE_CUDA
#  undef SS_BLOCKSIZE_CUDA
#  define SS_BLOCKSIZE_CUDA (32767)
# endif
#else
# define SS_BLOCKSIZE_CUDA (1024)
#endif
/* minstacksize = log(SS_BLOCKSIZE_CUDA) / log(3) * 2 */
#if SS_BLOCKSIZE_CUDA == 0
# if defined(BUILD_DIVSUFSORT64)
#  define SS_MISORT_STACKSIZE_CUDA (96)
# else
#  define SS_MISORT_STACKSIZE_CUDA (64)
# endif
#elif SS_BLOCKSIZE_CUDA <= 4096
# define SS_MISORT_STACKSIZE_CUDA (16)
#else
# define SS_MISORT_STACKSIZE_CUDA (24)
#endif
#if defined(BUILD_DIVSUFSORT64)
# define SS_SMERGE_STACKSIZE_CUDA (64)
#else
# define SS_SMERGE_STACKSIZE_CUDA (32)
#endif
/* for trsort.c */
#define TR_INSERTIONSORT_THRESHOLD_CUDA (8)
#if defined(BUILD_DIVSUFSORT64)
# define TR_STACKSIZE_CUDA (96)
#else
# define TR_STACKSIZE_CUDA (64)
#endif


/*- Macros -*/
#ifndef SWAP_CUDA
# define SWAP_CUDA(_a, _b) do { t = (_a); (_a) = (_b); (_b) = t; } while(0)
#endif /* SWAP_CUDA */
#ifndef MIN_CUDA
# define MIN_CUDA(_a, _b) (((_a) < (_b)) ? (_a) : (_b))
#endif /* MIN_CUDA */
#ifndef MAX_CUDA
# define MAX_CUDA(_a, _b) (((_a) > (_b)) ? (_a) : (_b))
#endif /* MAX_CUDA */
#define STACK_PUSH_CUDA(_a, _b, _c, _d)\
  do {\
    assert(ssize < STACK_SIZE);\
    stack[ssize].a = (_a), stack[ssize].b = (_b),\
    stack[ssize].c = (_c), stack[ssize++].d = (_d);\
  } while(0)
#define STACK_PUSH_CUDA5(_a, _b, _c, _d, _e)\
  do {\
    assert(ssize < STACK_SIZE);\
    stack[ssize].a = (_a), stack[ssize].b = (_b),\
    stack[ssize].c = (_c), stack[ssize].d = (_d), stack[ssize++].e = (_e);\
  } while(0)
#define STACK_POP_CUDA(_a, _b, _c, _d)\
  do {\
    assert(0 <= ssize);\
    if(ssize == 0) { return; }\
    (_a) = stack[--ssize].a, (_b) = stack[ssize].b,\
    (_c) = stack[ssize].c, (_d) = stack[ssize].d;\
  } while(0)
#define STACK_POP_CUDA5(_a, _b, _c, _d, _e)\
  do {\
    assert(0 <= ssize);\
    if(ssize == 0) { return; }\
    (_a) = stack[--ssize].a, (_b) = stack[ssize].b,\
    (_c) = stack[ssize].c, (_d) = stack[ssize].d, (_e) = stack[ssize].e;\
  } while(0)
/* for divsufsort_cuda.cu */
#define BUCKET_A_CUDA(_c0) bucket_A[(_c0)]
#if ALPHABET_SIZE_CUDA == 256
#define BUCKET_B_CUDA(_c0, _c1) (bucket_B[((_c1) << 8) | (_c0)])
#define BUCKET_B_CUDASTAR(_c0, _c1) (bucket_B[((_c0) << 8) | (_c1)])
#else
#define BUCKET_B_CUDA(_c0, _c1) (bucket_B[(_c1) * ALPHABET_SIZE_CUDA + (_c0)])
#define BUCKET_B_CUDASTAR(_c0, _c1) (bucket_B[(_c0) * ALPHABET_SIZE_CUDA + (_c1)])
#endif


/*- Private Prototypes -*/
/* sssort.c */
__device__
void
sssort_cuda(const sauchar_t *Td, const saidx_t *PA,
       saidx_t *first, saidx_t *last,
       saidx_t *buf, saidx_t bufsize,
       saidx_t depth, saidx_t n, saint_t lastsuffix);
/* trsort.c */
__device__
void
trsort_cuda(saidx_t *ISA, saidx_t *SA, saidx_t n, saidx_t depth);


#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* _DIVSUFSORT_PRIVATE_H */