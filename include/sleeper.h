#pragma once
#ifdef WIN32
#include <windows.h> /* WinAPI */

/* Windows sleep in 100ns units */
inline BOOLEAN nanosleep_simple(LONGLONG ns)
{
  // Convert true nanoseconds to 100-nanosecond intervals
  LONGLONG hundred_ns_intervals = ns / 100;

  /* Create and set the high-resolution timer */
  HANDLE timer = CreateWaitableTimerEx(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
  if (!timer)
    return FALSE;

  LARGE_INTEGER li;
  li.QuadPart = -hundred_ns_intervals; // Negative to indicate relative time
  if (!SetWaitableTimer(timer, &li, 0, NULL, NULL, FALSE))
  {
    CloseHandle(timer);
    return FALSE;
  }

  WaitForSingleObject(timer, INFINITE); // Wait for the timer to expire
  CloseHandle(timer);                   // Clean up
  return TRUE;
}

#else
#include <time.h>

inline void nanosleep_simple(long ns)
{
  struct timespec req;

  // Convert nanoseconds into seconds and nanoseconds
  req.tv_sec = ns / 1000000000L;  // Seconds
  req.tv_nsec = ns % 1000000000L; // Nanoseconds

  // Call nanosleep with requested time, no need to store remaining time
  nanosleep(&req, NULL);
}
#endif