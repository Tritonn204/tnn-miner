#ifndef SHA_DETECT_H
#define SHA_DETECT_H

#include <stdint.h>
#include <string.h>

#if defined(_WIN32)

#ifdef __x86_64__
#include <cpuid.h>
#endif

static inline int has_sha_ni_support(void)
{
#if defined(__x86_64__)
  // Redirected output is normal for services, benchmarks and captured logs.
  // Never infer a probe-child role from console handles: that used to exit
  // every redirected Windows miner before main(). SHA-NI uses XMM registers,
  // whose OS support is mandatory on Windows x64; no AVX/XGETBV gate is needed.
  unsigned int eax, ebx, ecx, edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return 0;
  if ((ecx & (1u << 19)) == 0) return 0; // SSE4.1, used by the SHA path.
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
  return (ebx & (1u << 29)) != 0; // SHA extensions.
#else
  return 0; // Not x86_64
#endif
}

#elif defined(__unix__) || defined(__APPLE__)

#include <unistd.h>
#include <sys/wait.h>

#ifdef __x86_64__
#include <immintrin.h>

__attribute__((target("sha,sse4.1")))
static inline void sha_probe_child_unix(int write_fd)
{
    __m128i a = _mm_setzero_si128();
    __m128i b = _mm_setzero_si128();
    __m128i c = _mm_setzero_si128();
    __m128i r = _mm_sha256rnds2_epu32(a, b, c);
    volatile uint32_t dummy = _mm_extract_epi32(r, 0);
    (void)dummy;

    write(write_fd, "1", 1);
    _exit(0);
}
#endif

static inline int has_sha_ni_support(void)
{
#if defined(__x86_64__)
  int pipefd[2];
  if (pipe(pipefd) != 0) return 0;

  pid_t pid = fork();
  if (pid < 0) return 0;

  if (pid == 0) {
    close(pipefd[0]);
    sha_probe_child_unix(pipefd[1]);
  } else {
    close(pipefd[1]);
    char result = 0;
    read(pipefd[0], &result, 1);
    close(pipefd[0]);
    waitpid(pid, NULL, 0);
    return result == '1';
  }
#else
  return 0; // Not x86_64
#endif
}

#else

static inline int has_sha_ni_support(void) { return 0; }

#endif

// Cached version
static inline int has_sha_ni_support_cached(void)
{
  static int cached = -1;
  if (cached == -1)
    cached = has_sha_ni_support();
  return cached;
}

#endif // SHA_DETECT_H
