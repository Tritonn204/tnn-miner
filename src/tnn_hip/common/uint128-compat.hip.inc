#pragma once

#if defined(__HIP_PLATFORM_AMD__) || (defined(__CUDA_ARCH__) && defined(__linux__))
    #define HAS_NATIVE_UINT128 1
    typedef __uint128_t uint128_t;
    
    __device__ __forceinline__ uint128_t make_uint128(uint64_t hi, uint64_t lo) {
        return ((__uint128_t)hi << 64) | lo;
    }
    
    __device__ __forceinline__ uint64_t uint128_low(uint128_t x) {
        return (uint64_t)x;
    }
    
    __device__ __forceinline__ uint64_t uint128_high(uint128_t x) {
        return (uint64_t)(x >> 64);
    }
    
    __device__ __forceinline__ uint128_t uint128_mul64(uint64_t a, uint64_t b) {
        return (__uint128_t)a * b;
    }
    
    __device__ __forceinline__ uint64_t uint128_div64(const uint128_t& dividend, uint64_t divisor) {
        return (uint64_t)(dividend / divisor);
    }
    
    __device__ __forceinline__ uint64_t uint128_mod64(const uint128_t& dividend, uint64_t divisor) {
        return (uint64_t)(dividend % divisor);
    }
    
    __device__ __forceinline__ bool uint128_less_than(const uint128_t& a, const uint128_t& b) {
        return a < b;
    }
    
    __device__ __forceinline__ bool uint128_greater_than(const uint128_t& a, const uint128_t& b) {
        return a > b;
    }
    
    __device__ __forceinline__ uint128_t uint128_sub(const uint128_t& a, const uint128_t& b) {
        return a - b;
    }
    
#else
    // Emulated 128-bit for NVIDIA on Windows
    #define HAS_NATIVE_UINT128 0
    
    struct uint128_t {
        uint64_t lo;
        uint64_t hi;
    };
    
    __device__ __forceinline__ uint128_t make_uint128(uint64_t hi, uint64_t lo) {
        uint128_t result;
        result.lo = lo;
        result.hi = hi;
        return result;
    }
    
    __device__ __forceinline__ uint64_t uint128_low(const uint128_t& x) {
        return x.lo;
    }
    
    __device__ __forceinline__ uint64_t uint128_high(const uint128_t& x) {
        return x.hi;
    }
    
    // Forward declarations for complex operations
    __device__ uint128_t uint128_mul64(uint64_t a, uint64_t b);
    __device__ uint64_t uint128_div64(const uint128_t& dividend, uint64_t divisor);
    __device__ uint64_t uint128_mod64(const uint128_t& dividend, uint64_t divisor);
    __device__ bool uint128_less_than(const uint128_t& a, const uint128_t& b);
    __device__ bool uint128_greater_than(const uint128_t& a, const uint128_t& b);
    __device__ uint128_t uint128_sub(const uint128_t& a, const uint128_t& b);
    
    // Implementations of complex operations
    __device__ inline uint128_t uint128_mul64(uint64_t a, uint64_t b) {
        // Split into 32-bit parts for accurate multiplication
        uint32_t a_lo = (uint32_t)a;
        uint32_t a_hi = (uint32_t)(a >> 32);
        uint32_t b_lo = (uint32_t)b;
        uint32_t b_hi = (uint32_t)(b >> 32);
        
        uint64_t p0 = (uint64_t)a_lo * b_lo;
        uint64_t p1 = (uint64_t)a_lo * b_hi;
        uint64_t p2 = (uint64_t)a_hi * b_lo;
        uint64_t p3 = (uint64_t)a_hi * b_hi;
        
        uint64_t middle = p1 + p2;
        uint64_t carry = (middle < p1) ? (1ULL << 32) : 0;
        
        uint128_t result;
        result.lo = p0 + (middle << 32);
        result.hi = p3 + (middle >> 32) + carry + (result.lo < p0 ? 1 : 0);
        return result;
    }
    
    __device__ inline uint64_t uint128_div64(const uint128_t& dividend, uint64_t divisor) {
        if (divisor == 0) return 0;
        
        uint64_t dividend_hi = dividend.hi;
        uint64_t dividend_lo = dividend.lo;
        
        // Special case: high part is 0
        if (dividend_hi == 0) {
            return dividend_lo / divisor;
        }
        
        // Use long division algorithm
        if (dividend_hi < divisor) {
            // Single precision division path - use floating point approximation
            double divisor_inv = 1.0 / (double)divisor;
            double dividend_d = (double)dividend_hi * 18446744073709551616.0 + (double)dividend_lo;
            uint64_t q_approx = (uint64_t)(dividend_d * divisor_inv);
            
            // Refine the quotient
            uint128_t prod = uint128_mul64(q_approx, divisor);
            while (prod.hi > dividend_hi || (prod.hi == dividend_hi && prod.lo > dividend_lo)) {
                q_approx--;
                uint128_t sub = uint128_mul64(1, divisor);
                prod.lo -= sub.lo;
                prod.hi -= sub.hi + (prod.lo > ~sub.lo ? 1 : 0);
            }
            
            return q_approx;
        } else {
            // Two-stage division
            uint64_t q_hi = dividend_hi / divisor;
            uint64_t r_hi = dividend_hi % divisor;
            
            // Now divide (r_hi:dividend_lo) by divisor
            double divisor_inv = 1.0 / (double)divisor;
            double dividend2_d = (double)r_hi * 18446744073709551616.0 + (double)dividend_lo;
            uint64_t q_lo = (uint64_t)(dividend2_d * divisor_inv);
            
            // Refine
            uint128_t check = make_uint128(r_hi, dividend_lo);
            uint128_t prod = uint128_mul64(q_lo, divisor);
            while (prod.hi > check.hi || (prod.hi == check.hi && prod.lo > check.lo)) {
                q_lo--;
                prod = uint128_mul64(q_lo, divisor);
            }
            
            return q_lo;
        }
    }
        
    __device__ inline uint64_t uint128_mod64(const uint128_t& dividend, uint64_t divisor) {
        if (divisor == 0) return 0;
        
        uint64_t dividend_hi = dividend.hi;
        uint64_t dividend_lo = dividend.lo;
        
        if (dividend_hi == 0) {
            return dividend_lo % divisor;
        }
        
        while (dividend_hi >= divisor) {
            dividend_hi = dividend_hi % divisor;
        }
        
        uint64_t q = uint128_div64(make_uint128(dividend_hi, dividend_lo), divisor);
        uint128_t prod = uint128_mul64(q, divisor);
        return dividend_lo - prod.lo;
    }
    
    __device__ inline bool uint128_less_than(const uint128_t& a, const uint128_t& b) {
        return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
    }
    
    __device__ inline bool uint128_greater_than(const uint128_t& a, const uint128_t& b) {
        return (a.hi > b.hi) || (a.hi == b.hi && a.lo > b.lo);
    }
    
    __device__ inline uint128_t uint128_sub(const uint128_t& a, const uint128_t& b) {
        uint128_t result;
        result.lo = a.lo - b.lo;
        result.hi = a.hi - b.hi - (a.lo < b.lo ? 1 : 0);
        return result;
    }
    
#endif // HAS_NATIVE_UINT128