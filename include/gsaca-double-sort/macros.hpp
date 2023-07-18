#pragma once

#define gsaca_always_inline __attribute__((always_inline)) inline
#define gsaca_likely(x) __builtin_expect(!!(x), 1)
#define gsaca_unlikely(x) __builtin_expect(!!(x), 0)