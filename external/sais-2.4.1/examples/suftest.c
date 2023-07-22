/*
 * suftest.c for sais
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


/* Checks the suffix array SA of the string T. */
static
int
sufcheck(const sa_uint8_t *T, const sa_int32_t *SA, sa_int32_t n, int verbose) {
  sa_int32_t C[256];
  sa_int32_t i, p, q, t;
  sa_uint8_t c;

  if(verbose) { fprintf(stderr, "sufcheck: "); }
  if(n == 0) {
    if(verbose) { fprintf(stderr, "Done.\n"); }
    return 0;
  }

  /* Check arguments. */
  if((T == NULL) || (SA == NULL) || (n < 0)) {
    if(verbose) { fprintf(stderr, "Invalid arguments.\n"); }
    return -1;
  }

  /* check range: [0..n-1] */
  for(i = 0; i < n; ++i) {
    if((SA[i] < 0) || (n <= SA[i])) {
      if(verbose) {
        fprintf(stderr, "Out of the range [0,%" SA_PRIdINT32 "].\n"
                        "  SA[%" SA_PRIdINT32 "]=%" SA_PRIdINT32 "\n",
                        n - 1, i, SA[i]);
      }
      return -2;
    }
  }

  /* check first characters. */
  for(i = 1; i < n; ++i) {
    if(T[SA[i - 1]] > T[SA[i]]) {
      if(verbose) {
        fprintf(stderr, "Suffixes in wrong order.\n"
                        "  T[SA[%" SA_PRIdINT32 "]=%" SA_PRIdINT32 "]=%d"
                        " > T[SA[%" SA_PRIdINT32 "]=%" SA_PRIdINT32 "]=%d\n",
                        i - 1, SA[i - 1], T[SA[i - 1]], i, SA[i], T[SA[i]]);
      }
      return -3;
    }
  }

  /* check suffixes. */
  for(i = 0; i < 256; ++i) { C[i] = 0; }
  for(i = 0; i < n; ++i) { ++C[T[i]]; }
  for(i = 0, p = 0; i < 256; ++i) {
    t = C[i];
    C[i] = p;
    p += t;
  }

  q = C[T[n - 1]];
  C[T[n - 1]] += 1;
  for(i = 0; i < n; ++i) {
    p = SA[i];
    if(0 < p) {
      c = T[--p];
      t = C[c];
    } else {
      c = T[p = n - 1];
      t = q;
    }
    if((t < 0) || (p != SA[t])) {
      if(verbose) {
        fprintf(stderr, "Suffix in wrong position.\n"
                        "  SA[%" SA_PRIdINT32 "]=%" SA_PRIdINT32 " or\n"
                        "  SA[%" SA_PRIdINT32 "]=%" SA_PRIdINT32 "\n",
                        t, (0 <= t) ? SA[t] : -1, i, SA[i]);
      }
      return -4;
    }
    if(t != q) {
      ++C[c];
      if((n <= C[c]) || (T[SA[C[c]]] != c)) { C[c] = -1; }
    }
  }

  if(1 <= verbose) { fprintf(stderr, "Done.\n"); }
  return 0;
}

static
void
print_help(const char *progname, int status) {
  fprintf(stderr, "usage: %s FILE\n\n", progname);
  exit(status);
}

int
main(int argc, const char *argv[]) {
  FILE *fp;
  const char *fname;
  sa_uint8_t *T;
  sa_int32_t *SA;
  LFS_OFF_T n;
  clock_t start, finish;
  int needclose = 1;

  /* Check arguments. */
  if((argc == 1) ||
     (strcmp(argv[1], "-h") == 0) ||
     (strcmp(argv[1], "--help") == 0)) { print_help(argv[0], EXIT_SUCCESS); }
  if(argc != 2) { print_help(argv[0], EXIT_FAILURE); }

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
    needclose = 0;
  }

  /* Get the file size. */
  if(LFS_FSEEK(fp, 0, SEEK_END) != 0) {
    fprintf(stderr, "%s: Could not fseek `%s': ", argv[0], fname);
    perror(NULL);
    exit(EXIT_FAILURE);
  }
  if((n = LFS_FTELL(fp)) < 0) {
    fprintf(stderr, "%s: Could not ftell `%s': ", argv[0], fname);
    perror(NULL);
    exit(EXIT_FAILURE);
  }
  if(0x7fffffff <= n) {
    fprintf(stderr, "%s: Input file `%s' is too big.\n", argv[0], fname);
    exit(EXIT_FAILURE);
  }
  rewind(fp);

  /* Allocate 5n bytes of memory. */
  T  = (sa_uint8_t *)malloc((size_t)n * sizeof(sa_uint8_t));
  SA = (sa_int32_t *)malloc((size_t)n * sizeof(sa_int32_t));
  if((T == NULL) || (SA == NULL)) {
    fprintf(stderr, "%s: Could not allocate memory.\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  /* Read n bytes of data. */
  if(fread(T, sizeof(sa_uint8_t), (size_t)n, fp) != (size_t)n) {
    fprintf(stderr, "%s: %s `%s': ",
      argv[0],
      (ferror(fp) || !feof(fp)) ? "Could not read data from" : "Unexpected EOF in",
      argv[1]);
    perror(NULL);
    exit(EXIT_FAILURE);
  }
  if(needclose & 1) { fclose(fp); }

  /* Construct the suffix array. */
  fprintf(stderr, "%s: %" LFS_PRId " bytes ... ", fname, n);
  start = clock();
  if(sais_u8(T, SA, (sa_int32_t)n, 256) != 0) {
    fprintf(stderr, "%s: Could not allocate memory.\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  finish = clock();
  fprintf(stderr, "%.4f sec\n", (double)(finish - start) / (double)CLOCKS_PER_SEC);

  /* Check the suffix array. */
  if(sufcheck(T, SA, (sa_int32_t)n, 1) != 0) { exit(EXIT_FAILURE); }

  /* Deallocate memory. */
  free(SA);
  free(T);

  return 0;
}
