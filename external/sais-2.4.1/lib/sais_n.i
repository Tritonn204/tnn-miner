/*
 * sais_n.i for sais
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

#define CONCATENATE_AGAIN(a, b) a ## b
#define CONCATENATE(a, b) CONCATENATE_AGAIN(a, b)
#define getCounts    CONCATENATE(getCounts_, SAIS_TYPENAME)
#define getBuckets   CONCATENATE(getBuckets_, SAIS_TYPENAME)
#define LMSsort1     CONCATENATE(LMSsort1_, SAIS_TYPENAME)
#define LMSpostproc1 CONCATENATE(LMSpostproc1_, SAIS_TYPENAME)
#define LMSsort2     CONCATENATE(LMSsort2_, SAIS_TYPENAME)
#define LMSpostproc2 CONCATENATE(LMSpostproc2_, SAIS_TYPENAME)
#define induceSA     CONCATENATE(induceSA_, SAIS_TYPENAME)
#define computeBWT   CONCATENATE(computeBWT_, SAIS_TYPENAME)
#define sais_main    CONCATENATE(sais_main_, SAIS_TYPENAME)

/* find the start or end of each bucket */
static
void
getCounts(const sais_char_type *T, sais_index_type *C, sais_index_type n, sais_index_type k) {
  sais_index_type i;
  for(i = 0; i < k; ++i) { C[i] = 0; }
  for(i = 0; i < n; ++i) { ++C[chr(i)]; }
}
static
void
getBuckets(const sais_index_type *C, sais_index_type *B, sais_index_type k, sais_bool_type end) {
  sais_index_type i, sum = 0;
  if(end) { for(i = 0; i < k; ++i) { sum += C[i]; B[i] = sum; } }
  else { for(i = 0; i < k; ++i) { sum += C[i]; B[i] = sum - C[i]; } }
}

/* sort all type LMS suffixes */
static
void
LMSsort1(const sais_char_type *T, sais_index_type *SA,
         sais_index_type *C, sais_index_type *B,
         sais_index_type n, sais_index_type k) {
  sais_index_type *b, i, j;
  sais_char_type c0, c1;

  /* compute SAl */
  if(C == B) { getCounts(T, C, n, k); }
  getBuckets(C, B, k, 0); /* find starts of buckets */
  j = n - 1;
  b = SA + B[c1 = chr(j)];
  --j;
  *b++ = (chr(j) < c1) ? ~j : j;
  for(i = 0; i < n; ++i) {
    if(0 < (j = SA[i])) {
      assert(chr(j) >= chr(j + 1));
      if((c0 = chr(j)) != c1) { B[c1] = b - SA; b = SA + B[c1 = c0]; }
      assert(i < (b - SA));
      --j;
      *b++ = (chr(j) < c1) ? ~j : j;
      SA[i] = 0;
    } else if(j < 0) {
      SA[i] = ~j;
    }
  }
  /* compute SAs */
  if(C == B) { getCounts(T, C, n, k); }
  getBuckets(C, B, k, 1); /* find ends of buckets */
  for(i = n - 1, b = SA + B[c1 = 0]; 0 <= i; --i) {
    if(0 < (j = SA[i])) {
      assert(chr(j) <= chr(j + 1));
      if((c0 = chr(j)) != c1) { B[c1] = b - SA; b = SA + B[c1 = c0]; }
      assert((b - SA) <= i);
      --j;
      *--b = (chr(j) > c1) ? ~(j + 1) : j;
      SA[i] = 0;
    }
  }
}
static
sais_index_type
LMSpostproc1(const sais_char_type *T, sais_index_type *SA,
             sais_index_type n, sais_index_type m) {
  sais_index_type i, j, p, q, plen, qlen, name;
  sais_bool_type diff;
  sais_char_type c0, c1;

  /* compact all the sorted substrings into the first m items of SA
      2*m must be not larger than n (proveable) */
  assert(0 < n);
  for(i = 0; (p = SA[i]) < 0; ++i) { SA[i] = ~p; assert((i + 1) < n); }
  if(i < m) {
    for(j = i, ++i;; ++i) {
      assert(i < n);
      if((p = SA[i]) < 0) {
        SA[j++] = ~p; SA[i] = 0;
        if(j == m) { break; }
      }
    }
  }

  /* store the length of all substrings */
  i = n - 1; j = n - 1; c0 = chr(n - 1);
  do { c1 = c0; } while((0 <= --i) && ((c0 = chr(i)) >= c1));
  for(; 0 <= i;) {
    do { c1 = c0; } while((0 <= --i) && ((c0 = chr(i)) <= c1));
    if(0 <= i) {
      SA[m + ((i + 1) >> 1)] = j - i; j = i + 1;
      do { c1 = c0; } while((0 <= --i) && ((c0 = chr(i)) >= c1));
    }
  }

  /* find the lexicographic names of all substrings */
  for(i = 0, name = 0, q = n, qlen = 0; i < m; ++i) {
    p = SA[i], plen = SA[m + (p >> 1)], diff = 1;
    if((plen == qlen) && ((q + plen) < n)) {
      for(j = 0; (j < plen) && (chr(p + j) == chr(q + j)); ++j) { }
      if(j == plen) { diff = 0; }
    }
    if(diff != 0) { ++name, q = p, qlen = plen; }
    SA[m + (p >> 1)] = name;
  }

  return name;
}
static
void
LMSsort2(const sais_char_type *T, sais_index_type *SA,
         sais_index_type *C, sais_index_type *B, sais_index_type *D,
         sais_index_type n, sais_index_type k) {
  sais_index_type *b, i, j, t, d;
  sais_char_type c0, c1;
  assert(C != B);

  /* compute SAl */
  getBuckets(C, B, k, 0); /* find starts of buckets */
  j = n - 1;
  b = SA + B[c1 = chr(j)];
  --j;
  t = (chr(j) < c1);
  j += n;
  *b++ = (t & 1) ? ~j : j;
  for(i = 0, d = 0; i < n; ++i) {
    if(0 < (j = SA[i])) {
      if(n <= j) { d += 1; j -= n; }
      assert(chr(j) >= chr(j + 1));
      if((c0 = chr(j)) != c1) { B[c1] = b - SA; b = SA + B[c1 = c0]; }
      assert(i < (b - SA));
      --j;
      t = c0; t = (t << 1) | (chr(j) < c1);
      if(D[t] != d) { j += n; D[t] = d; }
      *b++ = (t & 1) ? ~j : j;
      SA[i] = 0;
    } else if(j < 0) {
      SA[i] = ~j;
    }
  }
  for(i = n - 1; 0 <= i; --i) {
    if(0 < SA[i]) {
      if(SA[i] < n) {
        SA[i] += n;
        for(j = i - 1; SA[j] < n; --j) { }
        SA[j] -= n;
        i = j;
      }
    }
  }

  /* compute SAs */
  getBuckets(C, B, k, 1); /* find ends of buckets */
  for(i = n - 1, d += 1, b = SA + B[c1 = 0]; 0 <= i; --i) {
    if(0 < (j = SA[i])) {
      if(n <= j) { d += 1; j -= n; }
      assert(chr(j) <= chr(j + 1));
      if((c0 = chr(j)) != c1) { B[c1] = b - SA; b = SA + B[c1 = c0]; }
      assert((b - SA) <= i);
      --j;
      t = c0; t = (t << 1) | (chr(j) > c1);
      if(D[t] != d) { j += n; D[t] = d; }
      *--b = (t & 1) ? ~(j + 1) : j;
      SA[i] = 0;
    }
  }
}
static
sais_index_type
LMSpostproc2(sais_index_type *SA, sais_index_type n, sais_index_type m) {
  sais_index_type i, j, d, name;

  /* compact all the sorted LMS substrings into the first m items of SA */
  assert(0 < n);
  for(i = 0, name = 0; (j = SA[i]) < 0; ++i) {
    j = ~j;
    if(n <= j) { name += 1; }
    SA[i] = j;
    assert((i + 1) < n);
  }
  if(i < m) {
    for(d = i, ++i;; ++i) {
      assert(i < n);
      if((j = SA[i]) < 0) {
        j = ~j;
        if(n <= j) { name += 1; }
        SA[d++] = j; SA[i] = 0;
        if(d == m) { break; }
      }
    }
  }
  if(name < m) {
    /* store the lexicographic names */
    for(i = m - 1, d = name + 1; 0 <= i; --i) {
      if(n <= (j = SA[i])) { j -= n; --d; }
      SA[m + (j >> 1)] = d;
    }
  } else {
    /* unset flags */
    for(i = 0; i < m; ++i) {
      if(n <= (j = SA[i])) { j -= n; SA[i] = j; }
    }
  }

  return name;
}

/* compute SA and BWT */
static
void
induceSA(const sais_char_type *T, sais_index_type *SA,
         sais_index_type *C, sais_index_type *B,
         sais_index_type n, sais_index_type k) {
  sais_index_type *b, i, j;
  sais_char_type c0, c1;
  /* compute SAl */
  if(C == B) { getCounts(T, C, n, k); }
  getBuckets(C, B, k, 0); /* find starts of buckets */
  j = n - 1;
  b = SA + B[c1 = chr(j)];
  *b++ = ((0 < j) && (chr(j - 1) < c1)) ? ~j : j;
  for(i = 0; i < n; ++i) {
    j = SA[i], SA[i] = ~j;
    if(0 < j) {
      --j;
      assert(chr(j) >= chr(j + 1));
      if((c0 = chr(j)) != c1) { B[c1] = b - SA; b = SA + B[c1 = c0]; }
      assert(i < (b - SA));
      *b++ = ((0 < j) && (chr(j - 1) < c1)) ? ~j : j;
    }
  }
  /* compute SAs */
  if(C == B) { getCounts(T, C, n, k); }
  getBuckets(C, B, k, 1); /* find ends of buckets */
  for(i = n - 1, b = SA + B[c1 = 0]; 0 <= i; --i) {
    if(0 < (j = SA[i])) {
      --j;
      assert(chr(j) <= chr(j + 1));
      if((c0 = chr(j)) != c1) { B[c1] = b - SA; b = SA + B[c1 = c0]; }
      assert((b - SA) <= i);
      *--b = ((j == 0) || (chr(j - 1) > c1)) ? ~j : j;
    } else {
      SA[i] = ~j;
    }
  }
}
static
sais_index_type
computeBWT(const sais_char_type *T, sais_index_type *SA,
           sais_index_type *C, sais_index_type *B,
           sais_index_type n, sais_index_type k) {
  sais_index_type *b, i, j, pidx = -1;
  sais_char_type c0, c1;
  /* compute SAl */
  if(C == B) { getCounts(T, C, n, k); }
  getBuckets(C, B, k, 0); /* find starts of buckets */
  j = n - 1;
  b = SA + B[c1 = chr(j)];
  *b++ = ((0 < j) && (chr(j - 1) < c1)) ? ~j : j;
  for(i = 0; i < n; ++i) {
    if(0 < (j = SA[i])) {
      --j;
      assert(chr(j) >= chr(j + 1));
      SA[i] = ~((sais_index_type)(c0 = chr(j)));
      if(c0 != c1) { B[c1] = b - SA; b = SA + B[c1 = c0]; }
      assert(i < (b - SA));
      *b++ = ((0 < j) && (chr(j - 1) < c1)) ? ~j : j;
    } else if(j != 0) {
      SA[i] = ~j;
    }
  }
  /* compute SAs */
  if(C == B) { getCounts(T, C, n, k); }
  getBuckets(C, B, k, 1); /* find ends of buckets */
  for(i = n - 1, b = SA + B[c1 = 0]; 0 <= i; --i) {
    if(0 < (j = SA[i])) {
      --j;
      assert(chr(j) <= chr(j + 1));
      SA[i] = (c0 = chr(j));
      if(c0 != c1) { B[c1] = b - SA; b = SA + B[c1 = c0]; }
      assert((b - SA) <= i);
      *--b = ((0 < j) && (chr(j - 1) > c1)) ? ~((sais_index_type)chr(j - 1)) : j;
    } else if(j != 0) {
      SA[i] = ~j;
    } else {
      pidx = i;
    }
  }
  return pidx;
}

/* find the suffix array SA of T[0..n-1] in {0..255}^n */
static
sais_index_type
sais_main(const sais_char_type *T, sais_index_type *SA,
          sais_index_type fs, sais_index_type n, sais_index_type k,
          sais_bool_type isbwt) {
  sais_index_type *C, *B, *D, *RA, *b;
  sais_index_type i, j, m, p, q, t, name, pidx = 0, newfs;
  unsigned int flags;
  sais_char_type c0, c1;

  assert((T != NULL) && (SA != NULL));
  assert((0 <= fs) && (0 < n) && (1 <= k));

  if(k <= MINBUCKETSIZE) {
    if((C = SAIS_MYMALLOC(k, sais_index_type)) == NULL) { return -2; }
    if(k <= fs) {
      B = SA + (n + fs - k);
      flags = 1;
    } else {
      if((B = SAIS_MYMALLOC(k, sais_index_type)) == NULL) { SAIS_MYFREE(C, k, sais_index_type); return -2; }
      flags = 3;
    }
  } else if(k <= fs) {
    C = SA + (n + fs - k);
    if(k <= (fs - k)) {
      B = C - k;
      flags = 0;
    } else if(k <= (MINBUCKETSIZE * 4)) {
      if((B = SAIS_MYMALLOC(k, sais_index_type)) == NULL) { return -2; }
      flags = 2;
    } else {
      B = C;
      flags = 8;
    }
  } else {
    if((C = B = SAIS_MYMALLOC(k, sais_index_type)) == NULL) { return -2; }
    flags = 4 | 8;
  }
  if((n <= SAIS_LMSSORT2_LIMIT) && (2 <= (n / k))) {
    if(flags & 1) { flags |= ((k * 2) <= (fs - k)) ? 32 : 16; }
    else if((flags == 0) && ((k * 2) <= (fs - k * 2))) { flags |= 32; }
  }

  /* stage 1: reduce the problem by at least 1/2
     sort all the LMS-substrings */
  getCounts(T, C, n, k); getBuckets(C, B, k, 1); /* find ends of buckets */
  for(i = 0; i < n; ++i) { SA[i] = 0; }
  b = &t; i = n - 1; j = n; m = 0; c0 = chr(n - 1);
  do { c1 = c0; } while((0 <= --i) && ((c0 = chr(i)) >= c1));
  for(; 0 <= i;) {
    do { c1 = c0; } while((0 <= --i) && ((c0 = chr(i)) <= c1));
    if(0 <= i) {
      *b = j; b = SA + --B[c1]; j = i; ++m;
      do { c1 = c0; } while((0 <= --i) && ((c0 = chr(i)) >= c1));
    }
  }

  if(1 < m) {
    if(flags & (16 | 32)) {
      if(flags & 16) {
        if((D = SAIS_MYMALLOC(k * 2, sais_index_type)) == NULL) {
          if(flags & (1 | 4)) { SAIS_MYFREE(C, k, sais_index_type); }
          if(flags & 2) { SAIS_MYFREE(B, k, sais_index_type); }
          return -2;
        }
      } else {
        D = B - k * 2;
      }
      assert((j + 1) < n);
      ++B[chr(j + 1)];
      for(i = 0, j = 0; i < k; ++i) {
        j += C[i];
        if(B[i] != j) { assert(SA[B[i]] != 0); SA[B[i]] += n; }
        D[i] = D[i + k] = 0;
      }
      LMSsort2(T, SA, C, B, D, n, k);
      name = LMSpostproc2(SA, n, m);
      if(flags & 16) { SAIS_MYFREE(D, k * 2, sais_index_type); }
    } else {
      LMSsort1(T, SA, C, B, n, k);
      name = LMSpostproc1(T, SA, n, m);
    }
  } else if(m == 1) {
    *b = j + 1;
    name = 1;
  } else {
    name = 0;
  }

  /* stage 2: solve the reduced problem
     recurse if names are not yet unique */
  if(name < m) {
#ifdef IS_SAIS64
    sa_int32_t *SA32 = (sa_int32_t *)SA;
    if(flags & 4) { SAIS_MYFREE(C, k, sais_index_type); }
    if(flags & 2) { SAIS_MYFREE(B, k, sais_index_type); }
    newfs = (n + fs) - (m * 2);
    if((flags & (1 | 4)) == 0) {
      if((k + name) <= newfs) { newfs -= k; }
      else { flags |= 8; }
    }
    assert((n >> 1) <= (newfs + m));
    RA = SA + m + newfs;
    if(m < 0x7fffffff) {
      if(SA_UINT16_MAX < (name - 1)) { /* char_type = int32_t */
        sa_int32_t *RA32;
        newfs = (n + fs) * 2 - (m * 2);
        if((flags & (1 | 4)) == 0) {
          if((k + name) <= newfs) { newfs -= k; }
          else { flags |= 8; }
        }
        RA32 = SA32 + m + newfs;
        for(i = m + (n >> 1) - 1, j = m - 1; m <= i; --i) {
          if(SA[i] != 0) {
            assert((char *)(SA + i) <= (char *)(RA32 + j));
            RA32[j--] = (sa_int32_t)(SA[i] - 1);
          }
        }
        if(sais_main_i32(RA32, SA32, (sa_int32_t)newfs, (sa_int32_t)m, (sa_int32_t)name, 0) != 0) {
          if(flags & 1) { SAIS_MYFREE(C, k, sais_index_type); }
          return -2;
        }
      } else if(SA_UINT8_MAX < (name - 1)) { /* char_type = uint16_t */
        sa_uint16_t *RA16;
        newfs = (n + fs) * 2 - (m + (m * sizeof(sa_uint16_t) + sizeof(sa_int32_t) - 1) / sizeof(sa_int32_t));
        if((flags & (1 | 4)) == 0) {
          if((k + name) <= newfs) { newfs -= k; }
          else { flags |= 8; }
        }
        RA16 = (sa_uint16_t *)(SA32 + m + newfs);
        for(i = m + (n >> 1) - 1, j = m - 1; m <= i; --i) {
          if(SA[i] != 0) {
            assert((char *)(SA + i) <= (char *)(RA16 + j));
            RA16[j--] = (sa_uint16_t)(SA[i] - 1);
          }
        }
        if(sais_main_u16(RA16, SA32, (sa_int32_t)newfs, (sa_int32_t)m, (sa_int32_t)name, 0) != 0) {
          if(flags & 1) { SAIS_MYFREE(C, k, sais_index_type); }
          return -2;
        }
      } else { /* char_type = uint8_t */
        sa_uint8_t *RA8;
        newfs = (n + fs) * 2 - (m + (m * sizeof(sa_uint8_t) + sizeof(sa_int32_t) - 1) / sizeof(sa_int32_t));
        if((flags & (1 | 4)) == 0) {
          if((k + name) <= newfs) { newfs -= k; }
          else { flags |= 8; }
        }
        RA8 = (sa_uint8_t *)(SA32 + m + newfs);
        for(i = m + (n >> 1) - 1, j = m - 1; m <= i; --i) {
          if(SA[i] != 0) {
            assert((char *)(SA + i) <= (char *)(RA8 + j));
            RA8[j--] = (sa_uint8_t)(SA[i] - 1);
          }
        }
        if(sais_main_u8(RA8, SA32, (sa_int32_t)newfs, (sa_int32_t)m, (sa_int32_t)name, 0) != 0) {
          if(flags & 1) { SAIS_MYFREE(C, k, sais_index_type); }
          return -2;
        }
      }
    } else {
      if(SA_UINT32_MAX < (name - 1)) { /* char_type = int64_t */
        for(i = m + (n >> 1) - 1, j = m - 1; m <= i; --i) {
          if(SA[i] != 0) {
            RA[j--] = SA[i] - 1;
          }
        }
        if(sais_main_64_i64(RA, SA, newfs, m, name, 0) != 0) {
          if(flags & 1) { SAIS_MYFREE(C, k, sais_index_type); }
          return -2;
        }
      } else if(SA_UINT16_MAX < (name - 1)) { /* char_type = uint32_t */
        sa_uint32_t *RA32;
        newfs = (n + fs) - (m + (m * sizeof(sa_uint32_t) + sizeof(sa_int64_t) - 1) / sizeof(sa_int64_t));
        if((flags & (1 | 4)) == 0) {
          if((k + name) <= newfs) { newfs -= k; }
          else { flags |= 8; }
        }
        RA32 = (sa_uint32_t *)(SA + m + newfs);
        for(i = m + (n >> 1) - 1, j = m - 1; m <= i; --i) {
          if(SA[i] != 0) {
            assert((char *)(SA + i) <= (char *)(RA32 + j));
            RA32[j--] = (sa_uint32_t)(SA[i] - 1);
          }
        }
        if(sais_main_64_u32(RA32, SA, newfs, m, name, 0) != 0) {
          if(flags & 1) { SAIS_MYFREE(C, k, sais_index_type); }
          return -2;
        }
      } else if(SA_UINT8_MAX < (name - 1)) { /* char_type = uint16_t */
        sa_uint16_t *RA16;
        newfs = (n + fs) - (m + (m * sizeof(sa_uint16_t) + sizeof(sa_int64_t) - 1) / sizeof(sa_int64_t));
        if((flags & (1 | 4)) == 0) {
          if((k + name) <= newfs) { newfs -= k; }
          else { flags |= 8; }
        }
        RA16 = (sa_uint16_t *)(SA + m + newfs);
        for(i = m + (n >> 1) - 1, j = m - 1; m <= i; --i) {
          if(SA[i] != 0) {
            assert((char *)(SA + i) <= (char *)(RA16 + j));
            RA16[j--] = (sa_uint16_t)(SA[i] - 1);
          }
        }
        if(sais_main_64_u16(RA16, SA, newfs, m, name, 0) != 0) {
          if(flags & 1) { SAIS_MYFREE(C, k, sais_index_type); }
          return -2;
        }
      } else { /* char_type = uint8_t */
        sa_uint8_t *RA8;
        newfs = (n + fs) - (m + (m * sizeof(sa_uint8_t) + sizeof(sa_int64_t) - 1) / sizeof(sa_int64_t));
        if((flags & (1 | 4)) == 0) {
          if((k + name) <= newfs) { newfs -= k; }
          else { flags |= 8; }
        }
        RA8 = (sa_uint8_t *)(SA + m + newfs);
        for(i = m + (n >> 1) - 1, j = m - 1; m <= i; --i) {
          if(SA[i] != 0) {
            assert((char *)(SA + i) <= (char *)(RA8 + j));
            RA8[j--] = (sa_uint8_t)(SA[i] - 1);
          }
        }
        if(sais_main_64_u8(RA8, SA, newfs, m, name, 0) != 0) {
          if(flags & 1) { SAIS_MYFREE(C, k, sais_index_type); }
          return -2;
        }
      }
    }

    i = n - 1; j = m - 1; c0 = chr(n - 1);
    do { c1 = c0; } while((0 <= --i) && ((c0 = chr(i)) >= c1));
    for(; 0 <= i;) {
      do { c1 = c0; } while((0 <= --i) && ((c0 = chr(i)) <= c1));
      if(0 <= i) {
        RA[j--] = i + 1;
        do { c1 = c0; } while((0 <= --i) && ((c0 = chr(i)) >= c1));
      }
    }
    if(m < 0x7fffffff) {
      for(i = m - 1; 0 <= i; --i) { SA[i] = RA[SA32[i]]; }
    } else {
      for(i = 0; i < m; ++i) { SA[i] = RA[SA[i]]; }
    }
#else
    if(flags & 4) { SAIS_MYFREE(C, k, sais_index_type); }
    if(flags & 2) { SAIS_MYFREE(B, k, sais_index_type); }
    newfs = (n + fs) - (m * 2);
    if((flags & (1 | 4)) == 0) {
      if((k + name) <= newfs) { newfs -= k; }
      else { flags |= 8; }
    }
    RA = SA + m + newfs;
    if(SA_UINT16_MAX < (name - 1)) { /* char_type = int32_t */
      for(i = m + (n >> 1) - 1, j = m - 1; m <= i; --i) {
        if(SA[i] != 0) {
          assert((char *)(SA + i) <= (char *)(RA + j));
          RA[j--] = SA[i] - 1;
        }
      }
      if(sais_main_i32(RA, SA, newfs, m, name, 0) != 0) {
        if(flags & 1) { SAIS_MYFREE(C, k, sais_index_type); }
        return -2;
      }
    } else if(SA_UINT8_MAX < (name - 1)) { /* char_type = uint16_t */
      sa_uint16_t *RA16;
      newfs = (n + fs) - (m + (m * sizeof(sa_uint16_t) + sizeof(sa_int32_t) - 1) / sizeof(sa_int32_t));
      if((flags & (1 | 4)) == 0) {
        if((k + name) <= newfs) { newfs -= k; }
        else { flags |= 8; }
      }
      RA16 = (sa_uint16_t *)(SA + m + newfs);
      for(i = m + (n >> 1) - 1, j = m - 1; m <= i; --i) {
        if(SA[i] != 0) {
          assert((char *)(SA + i) <= (char *)(RA16 + j));
          RA16[j--] = (sa_uint16_t)(SA[i] - 1);
        }
      }
      if(sais_main_u16(RA16, SA, newfs, m, name, 0) != 0) {
        if(flags & 1) { SAIS_MYFREE(C, k, sais_index_type); }
        return -2;
      }
    } else { /* char_type = uint8_t */
      sa_uint8_t *RA8;
      newfs = (n + fs) - (m + (m * sizeof(sa_uint8_t) + sizeof(sa_int32_t) - 1) / sizeof(sa_int32_t));
      if((flags & (1 | 4)) == 0) {
        if((k + name) <= newfs) { newfs -= k; }
        else { flags |= 8; }
      }
      RA8 = (sa_uint8_t *)(SA + m + newfs);
      for(i = m + (n >> 1) - 1, j = m - 1; m <= i; --i) {
        if(SA[i] != 0) {
          assert((char *)(SA + i) <= (char *)(RA8 + j));
          RA8[j--] = (sa_uint8_t)(SA[i] - 1);
        }
      }
      if(sais_main_u8(RA8, SA, newfs, m, name, 0) != 0) {
        if(flags & 1) { SAIS_MYFREE(C, k, sais_index_type); }
        return -2;
      }
    }

    i = n - 1; j = m - 1; c0 = chr(n - 1);
    do { c1 = c0; } while((0 <= --i) && ((c0 = chr(i)) >= c1));
    for(; 0 <= i;) {
      do { c1 = c0; } while((0 <= --i) && ((c0 = chr(i)) <= c1));
      if(0 <= i) {
        RA[j--] = i + 1;
        do { c1 = c0; } while((0 <= --i) && ((c0 = chr(i)) >= c1));
      }
    }
    for(i = 0; i < m; ++i) { SA[i] = RA[SA[i]]; }
#endif
    if(flags & 4) {
      if((C = B = SAIS_MYMALLOC(k, sais_index_type)) == NULL) { return -2; }
    }
    if(flags & 2) {
      if((B = SAIS_MYMALLOC(k, sais_index_type)) == NULL) {
        if(flags & 1) { SAIS_MYFREE(C, k, sais_index_type); }
        return -2;
      }
    }
  }

  /* stage 3: induce the result for the original problem */
  if(flags & 8) { getCounts(T, C, n, k); }
  /* put all left-most S characters into their buckets */
  if(1 < m) {
    getBuckets(C, B, k, 1); /* find ends of buckets */
    i = m - 1, j = n, p = SA[m - 1], c1 = chr(p);
    do {
      q = B[c0 = c1];
      while(q < j) { SA[--j] = 0; }
      do {
        SA[--j] = p;
        if(--i < 0) { break; }
        p = SA[i];
      } while((c1 = chr(p)) == c0);
    } while(0 <= i);
    while(0 < j) { SA[--j] = 0; }
  }
  if(isbwt == 0) { induceSA(T, SA, C, B, n, k); }
  else { pidx = computeBWT(T, SA, C, B, n, k); }
  if(flags & (1 | 4)) { SAIS_MYFREE(C, k, sais_index_type); }
  if(flags & 2) { SAIS_MYFREE(B, k, sais_index_type); }

  return pidx;
}

#undef CONCATENATE_AGAIN
#undef CONCATENATE
#undef getCounts
#undef getBuckets
#undef stage1slowsort
#undef stage1fastsort
#undef induceSA
#undef computeBWT
#undef sais_main
