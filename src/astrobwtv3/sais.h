/*
 * sais.h for sais-lite
 * Copyright (c) 2008-2010 Yuta Mori All Rights Reserved.
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

#pragma once

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* find the suffix array SA of T[0..n-1]
   use a working space (excluding T and SA) of at most 2n+O(lg n) */
__device__ int
sais_hip(unsigned char *T, int *SA, int n);


// __device__ void sais_hip_s1( void *T, sais_index_type *SA,
//           sais_index_type fs, sais_index_type n, sais_index_type k, int cs,
//           sais_bool_type isbwt, workerData_hip &worker);

// __device__ void sais_hip_s2( void *T, sais_index_type *SA,
//           sais_index_type fs, sais_index_type n, sais_index_type k, int cs,
//           sais_bool_type isbwt, workerData_hip &worker);

// __device__ void sais_hip_s3( void *T, sais_index_type *SA,
//           sais_index_type fs, sais_index_type n, sais_index_type k, int cs,
//           sais_bool_type isbwt, workerData_hip &worker);


/* find the suffix array SA of T[0..n-1] in {0..k-1}^n
   use a working space (excluding T and SA) of at most MAX(4k,2n) */
__device__ int
sais_int_hip(int *T, int *SA, int n, int k);

/* burrows-wheeler transform */
__device__ int
sais_bwt(unsigned char *T, unsigned char *U, int *A, int n);
__device__ int
sais_int_bwt_hip(int *T, int *U, int *A, int n, int k);


#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */