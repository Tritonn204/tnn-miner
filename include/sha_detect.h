#ifndef SHA_DETECT_H
#define SHA_DETECT_H

#include <stdint.h>
#include <string.h>

#if defined(_WIN32)

#include <windows.h>

#ifdef __x86_64__
#include <immintrin.h>
#endif

static inline int has_sha_ni_support(void)
{
#if defined(__x86_64__)
  int supported = 0;

  HANDLE read_pipe, write_pipe;
  SECURITY_ATTRIBUTES sa = {sizeof(sa), NULL, TRUE};
  if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) return 0;

  char exe_path[MAX_PATH];
  if (!GetModuleFileNameA(NULL, exe_path, MAX_PATH)) return 0;

  STARTUPINFOA si = {sizeof(si)};
  PROCESS_INFORMATION pi;
  memset(&pi, 0, sizeof(pi));

  si.dwFlags |= STARTF_USESTDHANDLES;
  si.hStdOutput = write_pipe;
  si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
  si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);

  if (!CreateProcessA(NULL, exe_path, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
    CloseHandle(read_pipe);
    CloseHandle(write_pipe);
    return 0;
  }

  CloseHandle(write_pipe);
  WaitForSingleObject(pi.hProcess, INFINITE);

  char buf[1] = {0};
  DWORD read = 0;
  if (ReadFile(read_pipe, buf, 1, &read, NULL) && read == 1 && buf[0] == '1')
    supported = 1;

  CloseHandle(read_pipe);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);

  return supported;
#else
  return 0; // Not x86_64
#endif
}

#ifdef __x86_64__
__attribute__((constructor)) static void sha_probe_child_windows(void)
{
  HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
  if (hStdout == INVALID_HANDLE_VALUE) return;

  DWORD mode;
  if (!GetConsoleMode(hStdout, &mode)) {
    __m128i a = _mm_setzero_si128();
    __m128i b = _mm_setzero_si128();
    __m128i c = _mm_setzero_si128();
    __m128i r = _mm_sha256rnds2_epu32(a, b, c);
    volatile uint32_t dummy = _mm_extract_epi32(r, 0);
    (void)dummy;
    DWORD written;
    WriteFile(hStdout, "1", 1, &written, NULL);
    ExitProcess(0);
  }
}
#endif

#elif defined(__unix__) || defined(__APPLE__)

#include <unistd.h>
#include <sys/wait.h>

#ifdef __x86_64__
#include <immintrin.h>
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

    __m128i a = _mm_setzero_si128();
    __m128i b = _mm_setzero_si128();
    __m128i c = _mm_setzero_si128();
    __m128i r = _mm_sha256rnds2_epu32(a, b, c);
    volatile uint32_t dummy = _mm_extract_epi32(r, 0);
    (void)dummy;

    write(pipefd[1], "1", 1);
    _exit(0);
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
