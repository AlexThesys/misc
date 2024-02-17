#include "benchmark/benchmark.h"

#include <stdint.h>
#include <assert.h>
#include <memory>
#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <random>
#include <string.h>

typedef float fp32;
typedef uint8_t u8;
typedef uint16_t u16;
typedef int32_t s32;
typedef uint32_t u32;
typedef uint64_t u64;

constexpr u32 h = 1000;
constexpr u32 w = 800;

static constexpr u32 stride_avx = sizeof(__m256) / sizeof(fp32);
static constexpr u32 stride_sse = sizeof(__m128) / sizeof(fp32);

alignas(alignof(__m256i))
static const u32 rem_mask_table[stride_avx][stride_avx] = {
	{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
	{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xFFFFFFFF},
	{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF},
	{0x0, 0x0, 0x0, 0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
	{0x0, 0x0, 0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,},
	{0x0, 0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
	{0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,},
	{0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
};

extern "C" {
void __vec_mat_mul_linux64(fp32* res, const u16* tensor, const fp32* vector,u32 width, u32 height);
}

inline fp32 horizontal_add(const __m128& accum) {
	__m128 shuf = _mm_shuffle_ps(accum, accum, _MM_SHUFFLE(2, 3, 0, 1));
	__m128 sums = _mm_add_ps(accum, shuf);
	shuf = _mm_movehl_ps(shuf, sums);
	sums = _mm_add_ss(sums, shuf);
	return _mm_cvtss_f32(sums);
}

void __attribute__ ((noinline)) vec_mat_mul(fp32* res, const u16* tensor, const fp32* vector,u32 width, u32 height) {
	assert(0 == ((u64)vector & (sizeof(__m256) - 1)));
	const u32 rem = height & (stride_avx - 1);
	const u32 height_trunc = height - rem;
	const u32 rem_offset = height_trunc - (stride_avx - rem);
    const __m256i rem_mask = _mm256_load_si256((__m256i*)rem_mask_table[rem]);
    __m256 rem_vec = _mm256_maskload_ps(&vector[rem_offset], rem_mask);
    const u16* t_row = (u16*)tensor;
    for (u32 w = 0; w < width; w++, t_row += height) {
        fp32* out = &res[w];
        __m256 accum_256 = _mm256_setzero_ps();
        for (u32 h = 0; h < height_trunc; h += stride_avx) {
            __m128i t_ph = _mm_loadu_si128((__m128i*)(&t_row[h]));
            __m256 t_ps = _mm256_cvtph_ps(t_ph);
            t_ps = _mm256_mul_ps(t_ps, *(__m256*)(vector + h));
            accum_256 = _mm256_add_ps(accum_256, t_ps);
        }
        // compute reminder chunk
        __m128i t_ph = _mm_loadu_si128((__m128i*)(&t_row[rem_offset]));
        __m256 t_ps = _mm256_cvtph_ps(t_ph);
        t_ps = _mm256_mul_ps(t_ps, rem_vec);
        accum_256 = _mm256_add_ps(accum_256, t_ps);
        // horizontal add
        __m128 hi = _mm256_extractf128_ps(accum_256, 1);
        __m128 accum = _mm_add_ps(hi, *(__m128*) & accum_256);
        *out = horizontal_add(accum);
    }
}

void generate_data(u16* tensor, fp32* vector, u32 width, u32 height) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> fp_distr(0.0f, 1.0f);

	// fill vector
	for (u32 h = 0; h < height; h++) {
		vector[h] = fp_distr(gen);
	}
	alignas(alignof(__m128)) float ps[sizeof(__m128) / sizeof(fp32)];
	// fill tensor
	for (u32 w = 0; w < width; w++) {
		u16* t_row = (u16*)&tensor[w * height];
		for (u32 h = 0; h < height; h++) {
			ps[0] = fp_distr(gen);
			ps[1] = fp_distr(gen);
			ps[2] = fp_distr(gen);
			ps[3] = fp_distr(gen);
            __m128i ph = _mm_cvtps_ph(*(__m128*)ps, 0);
            _mm_storeu_si64((void*)&t_row[h], ph);
		}
	}
}


static void vec_mat_mul(benchmark::State& state) {
	const u32 height = h;
	const u32 width = w;

	alignas(alignof(__m256)) fp32 vector[h];
	u16* tensor = (u16*)malloc(w * h * sizeof(u16) + sizeof(__m256));
	generate_data(tensor, vector, w, h);
	alignas(alignof(__m256)) fp32 result_a[w];
    for (auto _ : state) {     
		vec_mat_mul(result_a, tensor, vector, width, height);
        benchmark::DoNotOptimize(result_a);
    }
	free(tensor);
}

static void _vec_mat_mul(benchmark::State& state) {
	const u32 height = h;
	const u32 width = w;

	alignas(alignof(__m256)) fp32 vector[h];
	u16* tensor = (u16*)malloc(w * h * sizeof(u16) + sizeof(__m256));
	generate_data(tensor, vector, w, h);
	alignas(alignof(__m256)) fp32 result_a[w];
    for (auto _ : state) {     
		__vec_mat_mul_linux64(result_a, tensor, vector, width, height);
        benchmark::DoNotOptimize(result_a);
    }
	free(tensor);
}

BENCHMARK(vec_mat_mul);
BENCHMARK(_vec_mat_mul);

BENCHMARK_MAIN();
