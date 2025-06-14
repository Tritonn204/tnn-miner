// This header is based on code from https://danlark.org/2020/06/14/128-bit-division/
// All credit goes to them for the logical flow of these methods

#include <cstdint>
#include <climits>
#include <iostream>

using tu_int = __uint128_t;
using du_int = uint64_t;
using su_int = uint32_t;

union utwords {
  tu_int all;
  struct {
    du_int low;
    du_int high;
  } s;
};

// Portable 128÷64 division (Hacker's Delight) if needed:
inline du_int Divide128div64to64Portable(du_int u1, du_int u0, du_int v, du_int* r);

// Arch-optimized version:
inline du_int Divide128div64to64(du_int u1, du_int u0, du_int v, du_int* r) {
#if defined(__x86_64__)
  du_int result;
  __asm__("divq %[v]"
          : "=a"(result), "=d"(*r)
          : [v] "r"(v), "a"(u0), "d"(u1));
  return result;
#else
  return Divide128div64to64Portable(u1, u0, v, r);
#endif
}

inline tu_int MyDivMod1(tu_int a, tu_int b, tu_int* rem) {
  utwords dividend{.all = a}, divisor{.all = b}, quotient{}, remainder{};
  const unsigned n_bits = sizeof(tu_int) * CHAR_BIT;

  // Fast path when divisor fits in 64 bits
  if (divisor.s.high == 0) {
    remainder.s.high = 0;
    if (dividend.s.high < divisor.s.low) {
      quotient.s.low = Divide128div64to64(dividend.s.high, dividend.s.low,
                                          divisor.s.low, &remainder.s.low);
      quotient.s.high = 0;
    } else {
      quotient.s.high =
          Divide128div64to64(0, dividend.s.high, divisor.s.low, &dividend.s.high);
      quotient.s.low = Divide128div64to64(dividend.s.high, dividend.s.low,
                                          divisor.s.low, &remainder.s.low);
    }
    if (rem) *rem = remainder.all;
    return quotient.all;
  }

  // Long division setup for full 128÷128
  int shift = __builtin_clzll(divisor.s.high) - __builtin_clzll(dividend.s.high);
  divisor.all <<= shift;
  quotient.s.low = 0;

  for (; shift >= 0; --shift) {
    quotient.s.low <<= 1;
    if (dividend.all >= divisor.all) {
      dividend.all -= divisor.all;
      quotient.s.low |= 1;
    }
    divisor.all >>= 1;
  }

  if (rem) *rem = dividend.all;
  return quotient.all;
}

struct MyDivision1 {
  tu_int value;
  MyDivision1() = default;
  MyDivision1(tu_int v) : value(v) {}
  MyDivision1(uint64_t high, uint64_t low)
      : value(((__uint128_t)high << 64) | low) {}

  friend MyDivision1 operator/(const MyDivision1& lhs, const MyDivision1& rhs) {
    tu_int q = MyDivMod1(lhs.value, rhs.value, nullptr);
    return MyDivision1(q);
  }

  friend MyDivision1 operator%(const MyDivision1& lhs, const MyDivision1& rhs) {
    tu_int r;
    MyDivMod1(lhs.value, rhs.value, &r);
    return MyDivision1(r);
  }

  friend std::ostream& operator<<(std::ostream& os, const MyDivision1& v) {
    char buf[50];
    tu_int n = v.value;
    char* p = buf + sizeof(buf);
    *--p = '\0';
    if (n == 0) *--p = '0';
    while (n > 0) {
      *--p = '0' + (n % 10);
      n /= 10;
    }
    return os << p;
  }

  uint64_t high64() const { return value >> 64; }
  uint64_t low64() const { return (uint64_t)value; }
};
