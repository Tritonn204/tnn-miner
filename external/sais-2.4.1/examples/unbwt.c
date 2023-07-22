/*
 * unbwt.c for sais
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

#ifdef HAVE_CONFIG_H
# include "sais_config.h"
#endif
#include <stdio.h>
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#else
# ifdef HAVE_MEMORY_H
#  include <memory.h>
# endif
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#else
# ifdef HAVE_STRINGS_H
#  include <strings.h>
# endif
#endif
#if defined(HAVE_IO_H) && defined(HAVE_FCNTL_H)
# include <io.h>
# include <fcntl.h>
#endif
#include <time.h>
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_INTTYPES_H
# include <inttypes.h>
#endif
#include "lfs.h"
#include "sais.h"


static
int
inverseBWT(const sa_uint8_t *T, sa_uint8_t *U, sa_int32_t *A, sa_int32_t n, sa_int32_t idx) {
  sa_int32_t C[256];
  sa_int32_t i, p;
  int c, len, half;

  /* Check arguments. */
  if((n < 0) || (idx < 0) || (n < idx) || ((0 < n) && (idx == 0))) { return -1; }
  if(n <= 1) { U[0] = T[0]; return 0; }

  /* Inverse BW transform. */
  for(c = 0; c < 256; ++c) { C[c] = 0; }
  for(i = 0; i < n; ++i) { ++C[T[i]]; }
  for(c = 0, i = 0; c < 256; ++c) {
    p = C[c];
    C[c] = i;
    i += p;
  }
  for(i = 0; i < idx; ++i) { A[C[T[i]]++] = i; }
  for(; i < n; ++i)        { A[C[T[i]]++] = i + 1; }
  for(i = 0, p = idx; i < n; ++i) {
    for(c = 0, len = 256, half = len >> 1;
        0 < len;
        len = half, half >>= 1) {
      if(C[c + half] < p) {
        c += half + 1;
        half -= (len & 1) ^ 1;
      }
    }
    U[i] = (sa_uint8_t)c;
    p = A[p - 1];
  }
  return 0;
}

static
size_t
read_int(FILE *fp, sa_int32_t *n) {
  sa_uint8_t c[4];
  size_t m = fread(c, sizeof(sa_uint8_t), 4, fp);
  if(m == 4) {
    *n = (c[0] <<  0) | (c[1] <<  8) |
         (c[2] << 16) | (c[3] << 24);
  }
  return m;
}
/*
static
int
read_vbcode(FILE *fp, sa_int32_t *n) {
  sa_uint8_t A[16];
  int i;
  sa_uint8_t c;
  *n = 0;
  if(fread(&c, sizeof(sa_uint8_t), 1, fp) != 1) { return 1; }
  i = c & 15;
  if(0 < i) {
    if(fread(&(A[0]), sizeof(sa_uint8_t), i, fp) != (size_t)i) { return 1; }
    while(0 < i--) { *n = (*n << 8) | A[i]; }
  }
  *n = (*n << 4) | ((c >> 4) & 15);
  return 0;
}
*/

static
void
print_help(const char *progname, int status) {
  fprintf(stderr, "usage: %s INFILE OUTFILE\n\n", progname);
  exit(status);
}

int
main(int argc, const char *argv[]) {
  FILE *fp, *ofp;
  const char *fname, *ofname;
  sa_uint8_t *T;
  sa_int32_t *A;
  LFS_OFF_T n;
  size_t m;
  clock_t start, finish;
  sa_int32_t idx, blocksize;
  int err, needclose = 3;

  /* Check arguments. */
  if((argc == 1) ||
     (strcmp(argv[1], "-h") == 0) ||
     (strcmp(argv[1], "--help") == 0)) { print_help(argv[0], EXIT_SUCCESS); }
  if(argc != 3) { print_help(argv[0], EXIT_FAILURE); }

  /* Open a file for reading. */
  if(strcmp(argv[1], "-") != 0) {
#ifdef HAVE_FOPEN_S
    if(fopen_s(&fp, fname = argv[1], "rb") != 0) {
#else
    if((fp = LFS_FOPEN(fname = argv[1], "rb")) == NULL) {
#endif
      fprintf(stderr, "%s: Could not open file `%s' for reading: ", argv[0], fname);
      perror(NULL);
      exit(EXIT_FAILURE);
    }
  } else {
#if defined(HAVE__SETMODE) && defined(HAVE__FILENO)
    if(_setmode(_fileno(stdin), _O_BINARY) == -1) {
      fprintf(stderr, "%s: Could not set mode: ", argv[0]);
      perror(NULL);
      exit(EXIT_FAILURE);
    }
#endif
    fp = stdin;
    fname = "stdin";
    needclose ^= 1;
  }

  /* Open a file for writing. */
  if(strcmp(argv[2], "-") != 0) {
#ifdef HAVE_FOPEN_S
    if(fopen_s(&ofp, ofname = argv[2], "wb") != 0) {
#else
    if((ofp = LFS_FOPEN(ofname = argv[2], "wb")) == NULL) {
#endif
      fprintf(stderr, "%s: Could not open file `%s' for writing: ", argv[0], ofname);
      perror(NULL);
      exit(EXIT_FAILURE);
    }
  } else {
#if defined(HAVE__SETMODE) && defined(HAVE__FILENO)
    if(_setmode(_fileno(stdout), _O_BINARY) == -1) {
      fprintf(stderr, "%s: Could not set mode: ", argv[0]);
      perror(NULL);
      exit(EXIT_FAILURE);
    }
#endif
    ofp = stdout;
    ofname = "stdout";
    needclose ^= 2;
  }

  /* Read the blocksize. */
  if(read_int(fp, &blocksize) != 4) {
    fprintf(stderr, "%s: Could not read data from `%s': ", argv[0], fname);
    perror(NULL);
    exit(EXIT_FAILURE);
  }

  /* Allocate 5n bytes of memory. */
  T = (sa_uint8_t *)malloc((size_t)(blocksize * sizeof(sa_uint8_t)));
  A = (sa_int32_t *)malloc((size_t)(blocksize * sizeof(sa_int32_t)));
  if((T == NULL) || (A == NULL)) {
    fprintf(stderr, "%s: Could not allocate memory.\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "UnBWT (blocksize %" SA_PRIdINT32 ") ... ", blocksize);
  start = clock();
  for(n = 0; (m = read_int(fp, &idx)) != 0; n += m) {
    /* Read blocksize bytes of data. */
    if((m != 4) || ((m = fread(T, sizeof(sa_uint8_t), (size_t)blocksize, fp)) == 0)) {
      fprintf(stderr, "%s: %s `%s': ",
        argv[0],
        (ferror(fp) || !feof(fp)) ? "Could not read data from" : "Unexpected EOF in",
        fname);
      perror(NULL);
      exit(EXIT_FAILURE);
    }

    /* Inverse Burrows-Wheeler Transform. */
    if((err = inverseBWT(T, T, A, (sa_int32_t)m, idx)) != 0) {
      fprintf(stderr, "%s (reverseBWT): %s.\n",
        argv[0],
        (err == -1) ? "Invalid data" : "Could not allocate memory");
      exit(EXIT_FAILURE);
    }

    /* Write m bytes of data. */
    if(fwrite(T, sizeof(sa_uint8_t), m, ofp) != m) {
      fprintf(stderr, "%s: Could not write data to `%s': ", argv[0], ofname);
      perror(NULL);
      exit(EXIT_FAILURE);
    }
  }
  if(ferror(fp)) {
    fprintf(stderr, "%s: Could not read data from `%s': ", argv[0], fname);
    perror(NULL);
    exit(EXIT_FAILURE);
  }
  finish = clock();
  fprintf(stderr, "%" LFS_PRId " bytes: %.4f sec\n",
    n, (double)(finish - start) / (double)CLOCKS_PER_SEC);

  /* Close files */
  if(needclose & 1) { fclose(fp); }
  if(needclose & 2) { fclose(ofp); }

  /* Deallocate memory. */
  free(A);
  free(T);

  return 0;
}
