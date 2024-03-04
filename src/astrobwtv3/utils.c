#include <inttypes.h>
#include <unistd.h>
#include <emmintrin.h>
#include <omp.h>

void
xor_buffer_aligned(unsigned char *buf1, unsigned char *buf2, size_t nblks, int blksz)
{
    uint64_t size, i;

    size = (uint64_t)nblks * (uint64_t)blksz;
    for (i = 0; i < size; i += blksz) {
        __m128i src1, src2, out;

        src1 = _mm_load_si128((__m128i *)buf1);
        src2 = _mm_load_si128((__m128i *)buf2);
        out = _mm_xor_si128(src1, src2);
        _mm_store_si128((__m128i *)buf1, out);
        buf1 += 16;  buf2 += 16;
        src1 = _mm_load_si128((__m128i *)buf1);
        src2 = _mm_load_si128((__m128i *)buf2);
        out = _mm_xor_si128(src1, src2);
        _mm_store_si128((__m128i *)buf1, out);
        buf1 += 16;  buf2 += 16;
        src1 = _mm_load_si128((__m128i *)buf1);
        src2 = _mm_load_si128((__m128i *)buf2);
        out = _mm_xor_si128(src1, src2);
        _mm_store_si128((__m128i *)buf1, out);
        buf1 += 16;  buf2 += 16;
        src1 = _mm_load_si128((__m128i *)buf1);
        src2 = _mm_load_si128((__m128i *)buf2);
        out = _mm_xor_si128(src1, src2);
        _mm_store_si128((__m128i *)buf1, out);
        buf1 += 16;  buf2 += 16;
    }
}