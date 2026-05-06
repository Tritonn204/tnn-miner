#pragma once

#include <cstdint>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#ifndef THREAD_PRIORITY_ABOVE_NORMAL
#define THREAD_PRIORITY_ABOVE_NORMAL 1
#endif
#else
#include <sched.h>
#ifndef THREAD_PRIORITY_ABOVE_NORMAL
#define THREAD_PRIORITY_ABOVE_NORMAL -5
#endif
#ifndef THREAD_PRIORITY_HIGHEST
#define THREAD_PRIORITY_HIGHEST -20
#endif
#ifndef THREAD_PRIORITY_TIME_CRITICAL
#define THREAD_PRIORITY_TIME_CRITICAL -20
#endif
#endif

void setAffinity(std::thread::native_handle_type t, uint64_t core);
void setPriority(std::thread::native_handle_type t, int priority);
