/*
 * sais.c for sais
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

#include "sais_config.h"
#include <assert.h>
#include <stdio.h>
#if HAVE_STDLIB_H
# include <stdlib.h>
#endif
#include "sais.h"

#define SA_UINT8_MAX (0xff)
#define SA_UINT16_MAX (0xffff)
#define MINBUCKETSIZE 256

static
sa_int32_t
sais_main_u8(const sa_uint8_t *T, sa_int32_t *SA, sa_int32_t fs, sa_int32_t n, sa_int32_t k, int isbwt);

static
sa_int32_t
sais_main_u16(const sa_uint16_t *T, sa_int32_t *SA, sa_int32_t fs, sa_int32_t n, sa_int32_t k, int isbwt);

static
sa_int32_t
sais_main_i32(const sa_int32_t *T, sa_int32_t *SA, sa_int32_t fs, sa_int32_t n, sa_int32_t k, int isbwt);

#ifndef SAIS_MYMALLOC
# define SAIS_MYMALLOC(_num, _type) ((_type *)malloc((_num) * sizeof(_type)))
#endif
#ifndef SAIS_MYFREE
# define SAIS_MYFREE(_ptr, _num, _type) free((_ptr))
#endif
#define chr(_a) T[(_a)]

#define sais_index_type sa_int32_t
#define sais_bool_type  int
#define SAIS_LMSSORT2_LIMIT SA_INT32_C(0x3fffffff)

/* 8-bit char_type */
#define SAIS_TYPENAME u8
#define sais_char_type  sa_uint8_t
#include "sais_n.i"
#undef SAIS_TYPENAME
#undef sais_char_type

/* 16-bit char_type */
#define SAIS_TYPENAME u16
#define sais_char_type  sa_uint16_t
#include "sais_n.i"
#undef SAIS_TYPENAME
#undef sais_char_type

/* 32-bit char_type */
#define SAIS_TYPENAME i32
#define sais_char_type  sa_int32_t
#include "sais_n.i"
#undef SAIS_TYPENAME
#undef sais_char_type


/*---------------------------------------------------------------------------*/

SAIS_DLLEXPORT
sa_int32_t
sais_u8(const sa_uint8_t *T, sa_int32_t *SA, sa_int32_t n, sa_int32_t k) {
  if((T == NULL) || (SA == NULL) || (n < 0) || (k <= 0) || (SA_UINT8_MAX < (k - 1))) { return -1; }
  if(n <= 1) { if(n == 1) { SA[0] = 0; } return 0; }
  return sais_main_u8(T, SA, 0, n, k, 0);
}

SAIS_DLLEXPORT
sa_int32_t
sais_u16(const sa_uint16_t *T, sa_int32_t *SA, sa_int32_t n, sa_int32_t k) {
  if((T == NULL) || (SA == NULL) || (n < 0) || (k <= 0) || (SA_UINT16_MAX < (k - 1))) { return -1; }
  if(n <= 1) { if(n == 1) { SA[0] = 0; } return 0; }
  return sais_main_u16(T, SA, 0, n, k, 0);
}

SAIS_DLLEXPORT
sa_int32_t
sais_i32(const sa_int32_t *T, sa_int32_t *SA, sa_int32_t n, sa_int32_t k) {
  if((T == NULL) || (SA == NULL) || (n < 0) || (k <= 0)) { return -1; }
  if(n <= 1) { if(n == 1) { SA[0] = 0; } return 0; }
  return sais_main_i32(T, SA, 0, n, k, 0);
}


/*---------------------------------------------------------------------------*/

SAIS_DLLEXPORT
sa_int32_t
sais_u8_bwt(const sa_uint8_t *T, sa_uint8_t *U, sa_int32_t *A, sa_int32_t n, sa_int32_t k) {
  sa_int32_t *B;
  sa_int32_t i, pidx;
  if((T == NULL) || (U == NULL) || (n < 0) || (k <= 0) || (SA_UINT8_MAX < (k - 1))) { return -1; }
  if(n <= 1) { if(n == 1) { U[0] = T[0]; } return n; }
  if((B = A) == NULL) { if((B = SAIS_MYMALLOC(n, sa_int32_t)) == NULL) { return -2; } }
  pidx = sais_main_u8(T, B, 0, n, k, 1);
  if(0 <= pidx) {
    U[0] = T[n - 1];
    for(i = 0; i < pidx; ++i) { U[i + 1] = (sa_uint8_t)B[i]; }
    for(i += 1; i < n; ++i) { U[i] = (sa_uint8_t)B[i]; }
    pidx += 1;
  }
  if(A == NULL) { SAIS_MYFREE(B, n, sa_int32_t); }
  return pidx;
}

SAIS_DLLEXPORT
sa_int32_t
sais_u16_bwt(const sa_uint16_t *T, sa_uint16_t *U, sa_int32_t *A, sa_int32_t n, sa_int32_t k) {
  sa_int32_t *B;
  sa_int32_t i, pidx;
  if((T == NULL) || (U == NULL) || (n < 0) || (k <= 0) || (SA_UINT16_MAX < (k - 1))) { return -1; }
  if(n <= 1) { if(n == 1) { U[0] = T[0]; } return n; }
  if((B = A) == NULL) { if((B = SAIS_MYMALLOC(n, sa_int32_t)) == NULL) { return -2; } }
  pidx = sais_main_u16(T, B, 0, n, k, 1);
  if(0 <= pidx) {
    U[0] = T[n - 1];
    for(i = 0; i < pidx; ++i) { U[i + 1] = (sa_uint16_t)B[i]; }
    for(i += 1; i < n; ++i) { U[i] = (sa_uint16_t)B[i]; }
    pidx += 1;
  }
  if(A == NULL) { SAIS_MYFREE(B, n, sa_int32_t); }
  return pidx;
}

SAIS_DLLEXPORT
sa_int32_t
sais_i32_bwt(const sa_int32_t *T, sa_int32_t *U, sa_int32_t *A, sa_int32_t n, sa_int32_t k) {
  sa_int32_t *B;
  sa_int32_t i, pidx;
  if((T == NULL) || (U == NULL) || (n < 0) || (k <= 0)) { return -1; }
  if(n <= 1) { if(n == 1) { U[0] = T[0]; } return n; }
  if((B = A) == NULL) { if((B = SAIS_MYMALLOC(n, sa_int32_t)) == NULL) { return -2; } }
  pidx = sais_main_i32(T, B, 0, n, k, 1);
  if(0 <= pidx) {
    U[0] = T[n - 1];
    for(i = 0; i < pidx; ++i) { U[i + 1] = B[i]; }
    for(i += 1; i < n; ++i) { U[i] = B[i]; }
    pidx += 1;
  }
  if(A == NULL) { SAIS_MYFREE(B, n, sa_int32_t); }
  return pidx;
}


/*---------------------------------------------------------------------------*/

SAIS_DLLEXPORT
const char *
sais_version(void) {
  return PROJECT_VERSION_FULL;
}
