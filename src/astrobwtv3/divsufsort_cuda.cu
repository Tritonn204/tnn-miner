/*
 * divsufsort.c for libdivsufsort
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

#include "divsufsort_def.cuh"
#include "divsufsort_private_cuda.cuh"

#include <cuda_runtime.h>


/*- Private Functions -*/

/* Sorts suffixes of type B*. */
__device__
saidx_t
sort_typeBstar_cuda(const sauchar_t *T, saidx_t *SA,
               saidx_t *bucket_A, saidx_t *bucket_B,
               saidx_t n) {
  saidx_t *PAb, *ISAb, *buf;
#ifdef _DSS_PARALLEL
  saidx_t *curbuf;
  saidx_t l;
#endif
  saidx_t k;
  saidx_t i, j, t, m, bufsize;
  saint_t c0, c1;
#ifdef _DSS_PARALLEL
  saint_t d0, d1;
  int tmp;
#endif

// For shared var reference only
// #pragma omp parallel default(shared) private(curbuf, k, l, d0, d1, tmp)

  #ifdef _DSS_PARALLEL
  // if (threadIdx.x % _DSS_THREADS == 0) {
  #endif
    /* Initialize bucket arrays. */
    for(i = 0; i < BUCKET_A_SIZE_CUDA; ++i) { bucket_A[i] = 0; }
    for(i = 0; i < BUCKET_B_SIZE_CUDA; ++i) { bucket_B[i] = 0; }

    /* Count the number of occurrences of the first one or two characters of each
      type A, B and B* suffix. Moreover, store the beginning position of all
      type B* suffixes into the array SA. */
    for(i = n - 1, m = n, c0 = T[n - 1]; 0 <= i;) {
      /* type A suffix. */
      do { ++BUCKET_A_CUDA(c1 = c0); } while((0 <= --i) && ((c0 = T[i]) >= c1));
      if(0 <= i) {
        /* type B* suffix. */
        ++BUCKET_B_CUDASTAR(c0, c1);
        SA[--m] = i;
        /* type B suffix. */
        for(--i, c1 = c0; (0 <= i) && ((c0 = T[i]) <= c1); --i, c1 = c0) {
          ++BUCKET_B_CUDA(c0, c1);
        }
      }
    }
    m = n - m;
  /*
  note:
    A type B* suffix is lexicographically smaller than a type B suffix that
    begins with the same first two characters.
  */

  /* Calculate the index of start/end point of each bucket. */
  
    #pragma unroll 256
    for(c0 = 0, i = 0, j = 0; c0 < ALPHABET_SIZE_CUDA; ++c0) {
      t = i + BUCKET_A_CUDA(c0);
      BUCKET_A_CUDA(c0) = i + j; /* start point */
      i = t + BUCKET_B_CUDA(c0, c0);
      #pragma unroll 256
      for(c1 = c0 + 1; c1 < ALPHABET_SIZE_CUDA; ++c1) {
        j += BUCKET_B_CUDASTAR(c0, c1);
        BUCKET_B_CUDASTAR(c0, c1) = j; /* end point */
        i += BUCKET_B_CUDA(c0, c1);
      }
    }
  #ifdef _DSS_PARALLEL
  // }
  // __syncwarp();
  #endif

  // printf("thread: %d, m = %d\n", threadIdx.x, m);
  if(0 < m) {
    /* Sort the type B* suffixes by their first two characters. */
    #ifdef _DSS_PARALLEL
    // if (threadIdx.x % _DSS_THREADS == 0) {
    #endif
      PAb = SA + n - m; ISAb = SA + m;
      for(i = m - 2; 0 <= i; --i) {
        t = PAb[i], c0 = T[t], c1 = T[t + 1];
        SA[--BUCKET_B_CUDASTAR(c0, c1)] = i;
      }
      t = PAb[m - 1], c0 = T[t], c1 = T[t + 1];
      SA[--BUCKET_B_CUDASTAR(c0, c1)] = m - 1;
    #ifdef _DSS_PARALLEL
    // }
    // __syncwarp();
    #endif

    /* Sort the type B* substrings using sssort. */
#ifdef _DSS_PARALLEL
    tmp = _DSS_THREADS;

    // if (threadIdx.x % _DSS_THREADS == 0){
      buf = SA + m, bufsize = (n - (2 * m)) / tmp;
      c0 = ALPHABET_SIZE_CUDA - 2, c1 = ALPHABET_SIZE_CUDA - 1, j = m;
    // }
    // __syncwarp();

// #pragma omp parallel default(shared) private(curbuf, k, l, d0, d1, tmp)
    // {
      tmp = threadIdx.x % _DSS_THREADS;
      curbuf = buf + tmp * bufsize;
      k = 0;

      // printf("thread: %d, tmp = %d\n", threadIdx.x, tmp);
      for(;;) {
        // if (threadIdx.x % _DSS_THREADS == 0) {
        // for(int tIndex = 0; tIndex < _DSS_THREADS; tIndex++) {
        //   if(threadIdx.x % _DSS_THREADS == tIndex) {
            // printf("tIndex = %d\n", tIndex);
            if(0 < (l = j)) {
              d0 = c0, d1 = c1;
              do {
                k = BUCKET_B_CUDASTAR(d0, d1);
                if(--d1 <= d0) {
                  d1 = ALPHABET_SIZE_CUDA - 1;
                  if(--d0 < 0) { break; }
                }
              } while(((l - k) <= 1) && (0 < (l = k)));
              c0 = d0, c1 = d1, j = k;
            }
          // }
        // }
        

      // printf("after loop\n");
        // }
      if(l == 0) { 
        // printf("thread: %d, l == 0\n", threadIdx.x);
        break; 
      }
      sssort_cuda(T, PAb, SA + k, SA + l,
              curbuf, bufsize, 2, n, *(SA + k) == (m - 1));
    }

    // printf("after loop\n");
    #ifdef _DSS_PARALLEL
    // __syncwarp();
    #endif
  // }
#else
  buf = SA + m, bufsize = n - (2 * m);
  for(c0 = ALPHABET_SIZE_CUDA - 2, j = m; 0 < j; --c0) {
    for(c1 = ALPHABET_SIZE_CUDA - 1; c0 < c1; j = i, --c1) {
      i = BUCKET_B_CUDASTAR(c0, c1);
      if(1 < (j - i)) {
        sssort_cuda(T, PAb, SA + i, SA + j,
                buf, bufsize, 2, n, *(SA + i) == (m - 1));
      }
    }
  }
#endif
    #ifdef _DSS_PARALLEL
    if (threadIdx.x % _DSS_THREADS == 0) {
    #endif
      /* Compute ranks of type B* substrings. */
      for(i = m - 1; 0 <= i; --i) {
        if(0 <= SA[i]) {
          j = i;
          do { ISAb[SA[i]] = i; } while((0 <= --i) && (0 <= SA[i]));
          SA[i + 1] = i - j;
          if(i <= 0) { break; }
        }
        j = i;
        do { ISAb[SA[i] = ~SA[i]] = j; } while(SA[--i] < 0);
        ISAb[SA[i]] = j;
      }

      /* Construct the inverse suffix array of type B* suffixes using trsort. */
      trsort_cuda(ISAb, SA, m, 1);

      /* Set the sorted order of tyoe B* suffixes. */
      for(i = n - 1, j = m, c0 = T[n - 1]; 0 <= i;) {
        for(--i, c1 = c0; (0 <= i) && ((c0 = T[i]) >= c1); --i, c1 = c0) { }
        if(0 <= i) {
          t = i;
          for(--i, c1 = c0; (0 <= i) && ((c0 = T[i]) <= c1); --i, c1 = c0) { }
          SA[ISAb[--j]] = ((t == 0) || (1 < (t - i))) ? t : ~t;
        }
      }

      /* Calculate the index of start/end point of each bucket. */
      BUCKET_B_CUDA(ALPHABET_SIZE_CUDA - 1, ALPHABET_SIZE_CUDA - 1) = n; /* end point */
      for(c0 = ALPHABET_SIZE_CUDA - 2, k = m - 1; 0 <= c0; --c0) {
        i = BUCKET_A_CUDA(c0 + 1) - 1;
        for(c1 = ALPHABET_SIZE_CUDA - 1; c0 < c1; --c1) {
          t = i - BUCKET_B_CUDA(c0, c1);
          BUCKET_B_CUDA(c0, c1) = i; /* end point */

          /* Move all type B* suffixes to the correct position. */
          for(i = t, j = BUCKET_B_CUDASTAR(c0, c1);
              j <= k;
              --i, --k) { SA[i] = SA[k]; }
        }
        BUCKET_B_CUDASTAR(c0, c0 + 1) = i - BUCKET_B_CUDA(c0, c0) + 1; /* start point */
        BUCKET_B_CUDA(c0, c0) = i; /* end point */
      }

    #ifdef _DSS_PARALLEL
    }
    // __syncwarp();
    #endif
  }

  return m;
}

/* Constructs the suffix array by using the sorted order of type B* suffixes. */
__device__
void
construct_SA_cuda(const sauchar_t *T, saidx_t *SA,
             saidx_t *bucket_A, saidx_t *bucket_B,
             saidx_t n, saidx_t m) {
  #ifdef _DSS_PARALLEL
  if (threadIdx.x % _DSS_THREADS == 0) {
  #endif  

    saidx_t *i, *j, *k;
    saidx_t s;
    saint_t c0, c1, c2;

    if(0 < m) {
      /* Construct the sorted order of type B suffixes by using
        the sorted order of type B* suffixes. */
      for(c1 = ALPHABET_SIZE_CUDA - 2; 0 <= c1; --c1) {
        /* Scan the suffix array from right to left. */
        for(i = SA + BUCKET_B_CUDASTAR(c1, c1 + 1),
            j = SA + BUCKET_A_CUDA(c1 + 1) - 1, k = NULL, c2 = -1;
            i <= j;
            --j) {
          if(0 < (s = *j)) {
            assert(T[s] == c1);
            assert(((s + 1) < n) && (T[s] <= T[s + 1]));
            assert(T[s - 1] <= T[s]);
            *j = ~s;
            c0 = T[--s];
            if((0 < s) && (T[s - 1] > c0)) { s = ~s; }
            if(c0 != c2) {
              if(0 <= c2) { BUCKET_B_CUDA(c2, c1) = k - SA; }
              k = SA + BUCKET_B_CUDA(c2 = c0, c1);
            }
            assert(k < j);
            *k-- = s;
          } else {
            assert(((s == 0) && (T[s] == c1)) || (s < 0));
            *j = ~s;
          }
        }
      }
    }
  

    /* Construct the suffix array by using
      the sorted order of type B suffixes. */
    k = SA + BUCKET_A_CUDA(c2 = T[n - 1]);
    *k++ = (T[n - 2] < c2) ? ~(n - 1) : (n - 1);
    /* Scan the suffix array from left to right. */
    for(i = SA, j = SA + n; i < j; ++i) {
      if(0 < (s = *i)) {
        assert(T[s - 1] >= T[s]);
        c0 = T[--s];
        if((s == 0) || (T[s - 1] < c0)) { s = ~s; }
        if(c0 != c2) {
          BUCKET_A_CUDA(c2) = k - SA;
          k = SA + BUCKET_A_CUDA(c2 = c0);
        }
        assert(i < k);
        *k++ = s;
      } else {
        assert(s < 0);
        *i = ~s;
      }
    }
  #ifdef _DSS_PARALLEL
  }
  #endif
}

/* Constructs the burrows-wheeler transformed string directly
   by using the sorted order of type B* suffixes. */
__device__
saidx_t
construct_BWT_cuda(const sauchar_t *T, saidx_t *SA,
              saidx_t *bucket_A, saidx_t *bucket_B,
              saidx_t n, saidx_t m) {
  saidx_t *i, *j, *k, *orig;
  saidx_t s;
  saint_t c0, c1, c2;

  if(0 < m) {
    /* Construct the sorted order of type B suffixes by using
       the sorted order of type B* suffixes. */
    for(c1 = ALPHABET_SIZE_CUDA - 2; 0 <= c1; --c1) {
      /* Scan the suffix array from right to left. */
      for(i = SA + BUCKET_B_CUDASTAR(c1, c1 + 1),
          j = SA + BUCKET_A_CUDA(c1 + 1) - 1, k = NULL, c2 = -1;
          i <= j;
          --j) {
        if(0 < (s = *j)) {
          assert(T[s] == c1);
          assert(((s + 1) < n) && (T[s] <= T[s + 1]));
          assert(T[s - 1] <= T[s]);
          c0 = T[--s];
          *j = ~((saidx_t)c0);
          if((0 < s) && (T[s - 1] > c0)) { s = ~s; }
          if(c0 != c2) {
            if(0 <= c2) { BUCKET_B_CUDA(c2, c1) = k - SA; }
            k = SA + BUCKET_B_CUDA(c2 = c0, c1);
          }
          assert(k < j);
          *k-- = s;
        } else if(s != 0) {
          *j = ~s;
#ifndef NDEBUG
        } else {
          assert(T[s] == c1);
#endif
        }
      }
    }
  }

  /* Construct the BWTed string by using
     the sorted order of type B suffixes. */
  k = SA + BUCKET_A_CUDA(c2 = T[n - 1]);
  *k++ = (T[n - 2] < c2) ? ~((saidx_t)T[n - 2]) : (n - 1);
  /* Scan the suffix array from left to right. */
  for(i = SA, j = SA + n, orig = SA; i < j; ++i) {
    if(0 < (s = *i)) {
      assert(T[s - 1] >= T[s]);
      c0 = T[--s];
      *i = c0;
      if((0 < s) && (T[s - 1] < c0)) { s = ~((saidx_t)T[s - 1]); }
      if(c0 != c2) {
        BUCKET_A_CUDA(c2) = k - SA;
        k = SA + BUCKET_A_CUDA(c2 = c0);
      }
      assert(i < k);
      *k++ = s;
    } else if(s != 0) {
      *i = ~s;
    } else {
      orig = i;
    }
  }

  return orig - SA;
}


/*---------------------------------------------------------------------------*/

/*- Function -*/
__device__
saint_t
divsufsort_cuda(const sauchar_t *T, saidx_t *SA, saidx_t n, saidx_t *bucket_A, saidx_t *bucket_B) {
  saidx_t m;
  saint_t err = 0;

  /* Check arguments. */
  if((T == NULL) || (SA == NULL) || (n < 0)) { return -1; }
  else if(n == 0) { return 0; }
  else if(n == 1) { SA[0] = 0; return 0; }
  else if(n == 2) { m = (T[0] < T[1]); SA[m ^ 1] = 0, SA[m] = 1; return 0; }

  /* Suffixsort. */
  if((bucket_A != NULL) && (bucket_B != NULL)) {
    m = sort_typeBstar_cuda(T, SA, bucket_A, bucket_B, n);
    construct_SA_cuda(T, SA, bucket_A, bucket_B, n, m);
  } else {
    err = -2;
  }

  return err;
}

__device__
saidx_t
divbwt_cuda(const sauchar_t *T, sauchar_t *U, saidx_t *A, saidx_t n) {
  saidx_t *B;
  saidx_t *bucket_A, *bucket_B;
  saidx_t m, pidx, i;

  /* Check arguments. */
  if((T == NULL) || (U == NULL) || (n < 0)) { return -1; }
  else if(n <= 1) { if(n == 1) { U[0] = T[0]; } return n; }

  if((B = A) == NULL) { B = (saidx_t *)malloc((size_t)(n + 1) * sizeof(saidx_t)); }

  /* Burrows-Wheeler Transform. */
  if((B != NULL) && (bucket_A != NULL) && (bucket_B != NULL)) {
    m = sort_typeBstar_cuda(T, B, bucket_A, bucket_B, n);
    pidx = construct_BWT_cuda(T, B, bucket_A, bucket_B, n, m);

    /* Copy to output string. */
    U[0] = T[n - 1];
    for(i = 0; i < pidx; ++i) { U[i + 1] = (sauchar_t)B[i]; }
    for(i += 1; i < n; ++i) { U[i] = (sauchar_t)B[i]; }
    pidx += 1;
  } else {
    pidx = -2;
  }

  free(bucket_B);
  free(bucket_A);
  if(A == NULL) { free(B); }

  return pidx;
}