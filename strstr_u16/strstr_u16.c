#include <stdint.h>
#include <string.h>
#include <intrin.h>
#include <assert.h>

typedef uint32_t u32;
typedef uint16_t u16;

/* str_sz has to be padded by step size */

#define _max(x, y) (x) > (y) ? (x) : (y)

#if defined(__AVX2__)
#include <immintrin.h>

#define step (sizeof(__m256i) / sizeof(u16))

static const u16* _strstr_u16_simd_(const u16* str, u32 str_sz, const u16* substr, u32 substr_sz) {
    assert(str_size >= _max((u32)step, substr_size));
    if (!substr_sz)
        return str;
    const __m256i first	= _mm256_set1_epi16(substr[0]);
    const __m256i last = _mm256_set1_epi16(substr[substr_sz - 1]);
    const u32 substr_sz_bytes = substr_sz * 2;
    const u32 skip_first = (u32)(substr_sz > 2);
    const u32 cmp_size = substr_sz_bytes - (2 << skip_first);

    for (u32 j = 0, sz = str_sz - substr_sz; j <= sz; j += step) {
        const u16 *f = str + j;
        const u16 *l = str + j + substr_sz - 1;
        __m256i ymm0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(f));
        __m256i ymm1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(l));

        ymm0 = _mm256_cmpeq_epi16(first, ymm0);
        ymm1 = _mm256_cmpeq_epi16(last, ymm1);
        ymm0 = _mm256_and_si256(ymm0, ymm1);

        u32 mask = _mm256_movemask_epi8(ymm0);

        const u32 max_offset = _min(step, str_sz - (j + substr_sz) + 1);
        const u32 max_offset_mask = (1 << (max_offset + max_offset)) - 1;
        constexpr u32 word_mask = 0x55555555;
        mask &= word_mask & max_offset_mask;
        unsigned long bit = 0;

        while (_BitScanForward(&bit, mask)) {
            const u32 offset = bit >> 1;
            const u16 *m0 = str + j + offset + skip_first;
            const u16 *m1 = substr + skip_first;
            if (memcmp(m0, m1, cmp_size) == 0)
                return (str + j + offset);

            mask ^= (1<<bit); // clear bit
        }

        //if (mask) {
        //	const u32 max_offset = _min(u32(step-1), str_sz - (j + substr_sz));
        //	for (u32 offset = 0, bit = 1; offset <= max_offset; offset++, bit <<= 2) {
        //		if (mask & bit) {
        //			const u16 *m0 = str + j + offset + skip_first;
        //			const u16 *m1 = substr + skip_first;
        //			if (memcmp(m0, m1, cmp_size) == 0)
        //				return str + j + offset;
        //		}
        //	}
        //}
    }

    return str + str_sz;
}
#else
#include <emmintrin.h>

#define step (sizeof(__m128i) / sizeof(u16))

static const u16* _strstr_u16_simd_(const u16* str, u32 str_sz, const u16* substr, u32 substr_sz) {
    assert(str_size >= _max((u32)step, substr_size));
    if (!substr_sz)
        return str;
    const __m128i first	= _mm_set1_epi16(substr[0]);
    const __m128i last = _mm_set1_epi16(substr[substr_sz - 1]);
    const u32 substr_sz_bytes = substr_sz * 2;
    const u32 skip_first = (u32)(substr_sz > 2);
    const u32 cmp_size = substr_sz_bytes - (2 << skip_first);

    for (u32 j = 0, sz = str_sz - substr_sz; j <= sz; j+= step) {
        const u16 *f = str + j;
        const u16 *l = str + j + substr_sz - 1;
        __m128i xmm0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(f));
        __m128i xmm1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(l));

        xmm0 = _mm_cmpeq_epi16(first, xmm0);
        xmm1 = _mm_cmpeq_epi16(last, xmm1);
        xmm0 = _mm_and_si128(xmm0, xmm1);

        u32 mask = (u32)_mm_movemask_epi8(xmm0);

        const u32 max_offset = _min(step, str_sz - (j + substr_sz) + 1);
        const u32 max_offset_mask = (1 << (max_offset + max_offset)) - 1;
        constexpr u32 word_mask = 0x00005555;
        mask &= word_mask & max_offset_mask;
        unsigned long bit = 0;

        while (_BitScanForward(&bit, mask)) {
            const u32 offset = bit >> 1;
            const u16 *m0 = str + j + offset + skip_first;
            const u16 *m1 = substr + skip_first;
            if (memcmp(m0, m1, cmp_size) == 0)
                return (str + j + offset);

            mask ^= (1<<bit); // clear bit
        }

        //if (mask) {
        //	const u32 max_offset = _min(u32(step-1), str_sz - (j + substr_sz));
        //	for (u32 offset = 0, bit = 1; offset <= max_offset; offset++, bit <<= 2) {
        //		if (mask & bit) {
        //			const u16 *m0 = str + j + offset + skip_first;
        //			const u16 *m1 = substr + skip_first;
        //			if (memcmp(m0, m1, cmp_size) == 0)
        //				return str + j + offset;
        //		}
        //	}
        //}
    }

    return str + str_sz;
}
#endif
