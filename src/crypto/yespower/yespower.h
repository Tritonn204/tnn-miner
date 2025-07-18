/*-
 * Copyright 2009 Colin Percival
 * Copyright 2013-2018 Alexander Peslyak
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * This file was originally written by Colin Percival as part of the Tarsnap
 * online backup system.
 */
#ifndef _YESPOWER_H_
#define _YESPOWER_H_

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus

#include <endian.hpp>
#include <boost/multiprecision/cpp_int.hpp>

using uint256_t = boost::multiprecision::uint256_t;
using cpp_dec_float_50 = boost::multiprecision::cpp_dec_float_50;

inline bool checkYespowerHash(uint8_t *hash_bytes, uint32_t *targetWords) {
    uint32_t *hash_words = (uint32_t*)hash_bytes;
    
    for (int i = 7; i >= 0; i--) {
        if (hash_words[i] > targetWords[i])
            return false;
        if (hash_words[i] < targetWords[i])
            return true;
    }
    
    return true;
}

extern "C" {
#endif

// Test result structure
typedef struct YespowerTestResult {
    bool matches;
    uint64_t ref_time_us;
    uint64_t fmv_time_us;
    char ref_hash[65];    // 32 bytes = 64 hex chars + null terminator
    char fmv_hash[65];    // 32 bytes = 64 hex chars + null terminator
} YespowerTestResult;

typedef struct {
    const uint8_t* data;
    size_t len;
} YespowerTestInput;

typedef struct {
    YespowerTestResult* results;
    size_t count;
    size_t capacity;
} YespowerTestResults;

typedef struct {
    const char *config_name;
    double ref_hash_rate;
    double opt_hash_rate;
    double speedup;
    int passed_correctness;
    uint64_t ref_time_us;
    uint64_t opt_time_us;
} yespower_bench_result_t;

/**
 * Internal type used by the memory allocator.  Please do not use it directly.
 * Use yespower_local_t instead.
 */
typedef struct {
	void *base, *aligned;
	size_t base_size, aligned_size;
} yespower_region_t;

/**
 * Type for thread-local (RAM) data structure.
 */
typedef yespower_region_t yespower_local_t;

/*
 * Type for yespower algorithm version numbers.
 */
typedef enum { YESPOWER_0_5 = 5, YESPOWER_1_0 = 10 } yespower_version_t;

/**
 * yespower parameters combined into one struct.
 */
typedef struct {
	yespower_version_t version;
	uint32_t N, r;
	const uint8_t *pers;
	size_t perslen;
} yespower_params_t;

/**
 * A 256-bit yespower hash.
 */
typedef struct {
	unsigned char uc[32];
} yespower_binary_t;

/**
 * yespower_init_local(local):
 * Initialize the thread-local (RAM) data structure.  Actual memory allocation
 * is currently fully postponed until a call to yespower().
 *
 * Return 0 on success; or -1 on error.
 *
 * MT-safe as long as local is local to the thread.
 */
extern int yespower_init_local(yespower_local_t *local);

/**
 * yespower_free_local(local):
 * Free memory that may have been allocated for an initialized thread-local
 * (RAM) data structure.
 *
 * Return 0 on success; or -1 on error.
 *
 * MT-safe as long as local is local to the thread.
 */
extern int yespower_free_local(yespower_local_t *local);

/**
 * yespower(local, src, srclen, params, dst):
 * Compute yespower(src[0 .. srclen - 1], N, r), to be checked for "< target".
 * local is the thread-local data structure, allowing to preserve and reuse a
 * memory allocation across calls, thereby reducing processing overhead.
 *
 * Return 0 on success; or -1 on error.
 *
 * local must be initialized with yespower_init_local().
 *
 * MT-safe as long as local and dst are local to the thread.
 */
extern int yespower(yespower_local_t *local,
    const uint8_t *src, size_t srclen,
    const yespower_params_t *params, yespower_binary_t *dst);

extern int yespower_ref(yespower_local_t *local,
    const uint8_t *src, size_t srclen,
    const yespower_params_t *params, yespower_binary_t *dst);

extern YespowerTestResult testYespower(const uint8_t* input, size_t input_len,
                               const yespower_params_t* params);

extern void runYespowerTestsC(YespowerTestResults* results);

/**
 * yespower_tls(src, srclen, params, dst):
 * Compute yespower(src[0 .. srclen - 1], N, r), to be checked for "< target".
 * The memory allocation is maintained internally using thread-local storage.
 *
 * Return 0 on success; or -1 on error.
 *
 * MT-safe as long as dst is local to the thread.
 */
extern int yespower_tls(const uint8_t *src, size_t srclen,
    const yespower_params_t *params, yespower_binary_t *dst);

extern int yespower_ref_tls(const uint8_t *src, size_t srclen,
    const yespower_params_t *params, yespower_binary_t *dst);

extern int yespower_tnn_tls(const uint8_t *src, size_t srclen,
    const yespower_params_t *params, yespower_binary_t *dst);

#ifdef __cplusplus
}

#include <vector>
void runYespowerTests(std::vector<YespowerTestResult>& results);
#endif

void print_yespower_benchmark_results(const yespower_bench_result_t *results, size_t num_results);
int benchmark_yespower_comparison(yespower_bench_result_t *results, size_t *num_results);
int benchmark_yespower_comparison_mt(yespower_bench_result_t *results, size_t *num_results);

#endif /* !_YESPOWER_H_ */