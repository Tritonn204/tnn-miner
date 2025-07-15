#ifndef SHA_DETECT_H
#define SHA_DETECT_H

#include <stdint.h>
#include <string.h>

#if defined(_WIN32)

#include <windows.h>
#include <immintrin.h>

static inline int has_sha_ni_support(void)
{
  int supported = 0;

  // Create anonymous pipe
  HANDLE read_pipe, write_pipe;
  SECURITY_ATTRIBUTES sa = {sizeof(sa), NULL, TRUE};
  if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0))
    return 0;

  // Get path to current executable
  char exe_path[MAX_PATH];
  if (!GetModuleFileNameA(NULL, exe_path, MAX_PATH))
    return 0;

  // Prepare child startup info
  STARTUPINFOA si = {sizeof(si)};
  PROCESS_INFORMATION pi;
  memset(&pi, 0, sizeof(pi));

  si.dwFlags |= STARTF_USESTDHANDLES;
  si.hStdOutput = write_pipe;
  si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

  // Launch child with no arguments
  if (!CreateProcessA(NULL, exe_path, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi))
  {
    CloseHandle(read_pipe);
    CloseHandle(write_pipe);
    return 0;
  }

  CloseHandle(write_pipe); // Parent doesn't write

  // Wait and read child response
  WaitForSingleObject(pi.hProcess, INFINITE);

  char buf[1] = {0};
  DWORD read = 0;
  if (ReadFile(read_pipe, buf, 1, &read, NULL) && read == 1 && buf[0] == '1')
    supported = 1;

  CloseHandle(read_pipe);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);

  return supported;
}

// Auto-run probe in child
__attribute__((constructor)) static void sha_probe_child_windows(void)
{
  DWORD ppid = GetCurrentProcessId(); // Can enhance with true parent ID if needed
  HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
  if (hStdout == INVALID_HANDLE_VALUE)
    return;

  DWORD mode;
  if (!GetConsoleMode(hStdout, &mode))
  {
    // We're likely the probe child (stdout is redirected)
    __m128i a = _mm_setzero_si128();
    __m128i b = _mm_setzero_si128();
    __m128i c = _mm_setzero_si128();
    __m128i r = _mm_sha256rnds2_epu32(a, b, c);
    volatile uint32_t dummy = _mm_extract_epi32(r, 0);
    (void)dummy;
    DWORD written;
    WriteFile(hStdout, "1", 1, &written, NULL);
    ExitProcess(0); // Do not run main()
  }
}

#elif defined(__unix__) || defined(__APPLE__)

#include <unistd.h>
#include <sys/wait.h>
#include <immintrin.h>

static inline int has_sha_ni_support(void)
{
  int pipefd[2];
  if (pipe(pipefd) != 0)
    return 0;

  pid_t pid = fork();
  if (pid < 0)
    return 0;

  if (pid == 0)
  {
    // Child: try SHA-NI, send success
    close(pipefd[0]);

    __m128i a = _mm_setzero_si128();
    __m128i b = _mm_setzero_si128();
    __m128i c = _mm_setzero_si128();
    __m128i r = _mm_sha256rnds2_epu32(a, b, c);
    volatile uint32_t dummy = _mm_extract_epi32(r, 0);
    (void)dummy;

    write(pipefd[1], "1", 1);
    _exit(0); // No return to main()
  }
  else
  {
    // Parent
    close(pipefd[1]);
    char result = 0;
    read(pipefd[0], &result, 1);
    close(pipefd[0]);
    waitpid(pid, NULL, 0);
    return result == '1';
  }
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