#pragma once

#include <stdint.h>

// This function is HOST-ONLY. Never call it from __device__ / kernels.
#ifdef _WIN32
    #include <windows.h>

    inline void nanosleep_simple(int64_t ns)
    {
        if (ns <= 0) return;

        // Convert true nanoseconds to 100-nanosecond intervals
        int64_t hundred_ns_intervals = ns / 100;

        // Fallback: if high-res timer creation fails, use Sleep(ms)
        auto fallback_sleep = [ns]() {
            DWORD ms = (DWORD)((ns + 999999) / 1000000); // round up
            ::Sleep(ms);
        };

        HANDLE timer = CreateWaitableTimerExW(
            nullptr,
            nullptr,
            CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
            TIMER_ALL_ACCESS
        );

        if (!timer) {
            fallback_sleep();
            return;
        }

        LARGE_INTEGER li;
        // Negative to indicate relative time
        li.QuadPart = -hundred_ns_intervals;

        if (!SetWaitableTimer(timer, &li, 0, nullptr, nullptr, FALSE)) {
            CloseHandle(timer);
            fallback_sleep();
            return;
        }

        WaitForSingleObject(timer, INFINITE);
        CloseHandle(timer);
    }

#else  // POSIX (Linux, etc.)

    #include <time.h>

    inline void nanosleep_simple(int64_t ns)
    {
        if (ns <= 0) return;

        struct timespec req;
        req.tv_sec  = ns / 1000000000LL;      // seconds
        req.tv_nsec = ns % 1000000000LL;      // nanoseconds

        nanosleep(&req, nullptr);
    }

#endif
