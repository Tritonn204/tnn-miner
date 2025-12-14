#pragma once

// limits.h
#if defined _WIN32 || defined _WIN64
 #define __WORDSIZE 32
#else
 #if defined __x86_64__ && !defined __ILP32__
  #define __WORDSIZE 64
 #else
  #define __WORDSIZE 32
 #endif
#endif
#define MB_LEN_MAX  16
#define CHAR_BIT    8
#define SCHAR_MIN   (-128)
#define SCHAR_MAX   127
#define UCHAR_MAX   255
enum {
  _JITIFY_CHAR_IS_UNSIGNED = (char)-1 >= 0,
  CHAR_MIN = _JITIFY_CHAR_IS_UNSIGNED ? 0 : SCHAR_MIN,
  CHAR_MAX = _JITIFY_CHAR_IS_UNSIGNED ? UCHAR_MAX : SCHAR_MAX,
};
#define SHRT_MIN    (-32768)
#define SHRT_MAX    32767
#define USHRT_MAX   65535
#define INT_MIN     (-INT_MAX - 1)
#define INT_MAX     2147483647
#define UINT_MAX    4294967295U
#if __WORDSIZE == 64
 # define LONG_MAX  9223372036854775807L
#else
 # define LONG_MAX  2147483647L
#endif
#define LONG_MIN    (-LONG_MAX - 1L)
#if __WORDSIZE == 64
 #define ULONG_MAX  18446744073709551615UL
#else
 #define ULONG_MAX  4294967295UL
#endif
#define LLONG_MAX  9223372036854775807LL
#define LLONG_MIN  (-LLONG_MAX - 1LL)
#define ULLONG_MAX 18446744073709551615ULL

// End of limits.h
typedef signed char      int8_t;
typedef signed short     int16_t;
typedef signed int       int32_t;
typedef signed long long int64_t;
typedef signed char      int_fast8_t;
typedef signed short     int_fast16_t;
typedef signed int       int_fast32_t;
typedef signed long long int_fast64_t;
typedef signed char      int_least8_t;
typedef signed short     int_least16_t;
typedef signed int       int_least32_t;
typedef signed long long int_least64_t;
typedef signed long long intmax_t;
typedef signed long      intptr_t; //optional
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned char      uint_fast8_t;
typedef unsigned short     uint_fast16_t;
typedef unsigned int       uint_fast32_t;
typedef unsigned long long uint_fast64_t;
typedef unsigned char      uint_least8_t;
typedef unsigned short     uint_least16_t;
typedef unsigned int       uint_least32_t;
typedef unsigned long long uint_least64_t;
typedef unsigned long long uintmax_t;
#define INT8_MIN    SCHAR_MIN
#define INT16_MIN   SHRT_MIN
#if defined _WIN32 || defined _WIN64
#define WCHAR_MIN   SHRT_MIN
#define WCHAR_MAX   SHRT_MAX
typedef unsigned long long uintptr_t; //optional
#else
#define WCHAR_MIN   INT_MIN
#define WCHAR_MAX   INT_MAX
typedef unsigned long      uintptr_t; //optional
#endif
#define INT32_MIN   INT_MIN
#define INT64_MIN   LLONG_MIN
#define INT8_MAX    SCHAR_MAX
#define INT16_MAX   SHRT_MAX
#define INT32_MAX   INT_MAX
#define INT64_MAX   LLONG_MAX
#define UINT8_MAX   UCHAR_MAX
#define UINT16_MAX  USHRT_MAX
#define UINT32_MAX  UINT_MAX
#define UINT64_MAX  ULLONG_MAX
#define INTPTR_MIN  LONG_MIN
#define INTMAX_MIN  LLONG_MIN
#define INTPTR_MAX  LONG_MAX
#define INTMAX_MAX  LLONG_MAX
#define UINTPTR_MAX ULONG_MAX
#define UINTMAX_MAX ULLONG_MAX
#define PTRDIFF_MIN INTPTR_MIN
#define PTRDIFF_MAX INTPTR_MAX
#define SIZE_MAX    UINT64_MAX