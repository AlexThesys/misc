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
typedef uint16_t fp16;
typedef int32_t s32;
typedef uint32_t u32;
typedef uint64_t u64;

static constexpr u32 stride_avx = sizeof(__m256) / sizeof(fp32);
static constexpr u32 stride_sse = sizeof(__m128) / sizeof(fp32);

inline fp32 horizontal_add(const __m128& accum) {
	__m128 shuf = _mm_shuffle_ps(accum, accum, _MM_SHUFFLE(2, 3, 0, 1));
	__m128 sums = _mm_add_ps(accum, shuf);
	shuf = _mm_movehl_ps(shuf, sums);
	sums = _mm_add_ss(sums, shuf);
	return _mm_cvtss_f32(sums);
}

void batch_mat_mul_transposed_6x6(const fp32* A, const fp32* B, fp32* Y, u32 batches_num) {
	constexpr u32 size = 6;
	constexpr u32 size_sq = size * size;
	for (u32 b = 0; b < batches_num; ++b, B += size_sq) {
		for (s32 r = 0; r < size; ++r, A += size) {
			for (s32 c = 0; c < size; ++c, ++Y) {
				fp32 sum = 0.f;
				for (s32 i = 0; i < size; ++i) {
					sum += A[i] * B[size * c + i];
				}
				*Y = sum;
			}
		}
	}
}

void batch_mat_mul_transposed_6x6_simd(const fp32* A, const fp32* B, fp32* Y, u32 batches_num) {
	constexpr u32 size = 6;
	constexpr u32 size_sq = size * size;
	const __m128i rem_mask_m128 = _mm_set_epi32(0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF);
#if defined(__AVX__) || defined(__AVX2__)
#	if defined(_M_FP_FAST) && !defined(_M_FP_EXCEPT)
	// we are reading past the end of the array and doing arithmetic in the last iteration, but with fp exceptions turned of it's fine 
	for (u32 i = 0; i < batches_num; ++i, A += size_sq, Y += size_sq) {
		for (s32 j = 0; j < size; ++j, B += size) {
			const __m256 b = _mm256_loadu_ps(B);
			for (s32 k = 0; k < size_sq; k += size) {
				__m256 a = _mm256_loadu_ps(A + k);
				a = _mm256_mul_ps(a, b);
				__m128 accum = _mm256_extractf128_ps(a, 1);
				accum = _mm_and_ps(accum, *(__m128*)&rem_mask_m128);
				accum = _mm_add_ps(accum, *(__m128*)&a);
				Y[k + j] = horizontal_add(accum);
			}
		}
	}
#	else
	const __m128i rem_mask_m128_inv = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0);
	for (u32 i = 0; i < batches_num - 1; ++i, A += size_sq, Y+=size_sq) {
		for (s32 j = 0; j < size; ++j, B += size) {
			const __m256 b = _mm256_loadu_ps(B);
			for (s32 k = 0; k < size_sq; k+=size) {
				__m256 a = _mm256_loadu_ps(A+k);
				a = _mm256_mul_ps(a, b);
				__m128 accum = _mm256_extractf128_ps(a, 1);
				accum = _mm_and_ps(accum, *(__m128*)&rem_mask_m128);
				accum = _mm_add_ps(accum, *(__m128*)&a);
				Y[k+j] = horizontal_add(accum);
			}
		}
	}
	// compute the last matrix in the batch
	for (s32 j = 0; j < size - 1; ++j, B += size) {
		const __m256 b = _mm256_loadu_ps(B);
		for (s32 k = 0; k < size_sq; k += size) {
			__m256 a = _mm256_load_ps(A+k);
			a = _mm256_mul_ps(a, b);
			__m128 accum = _mm256_extractf128_ps(a, 1);
			accum = _mm_and_ps(accum, *(__m128*) & rem_mask_m128);
			accum = _mm_add_ps(accum, *(__m128*)&a);
			Y[k + j] = horizontal_add(accum);
		}
	}
	// last vector in the last matrix
	const __m256 b = _mm256_loadu_ps(B - 2);
	A -= 2;
	for (s32 k = 0; k < size_sq; k += size) {
		__m256 a = _mm256_loadu_ps(A + k);
		a = _mm256_mul_ps(a, b);
		__m128 accum = _mm256_extractf128_ps(a, 1);
		*(__m128*)&a = _mm_and_ps(*(__m128*)&a, *(__m128*)&rem_mask_m128_inv);
		accum = _mm_add_ps(accum, *(__m128*)&a);
		Y[k + size - 1] = horizontal_add(accum);
	}
#	endif //  defined(_M_FP_FAST) && !defined(_M_FP_EXCEPT)
#else
#	if defined(_M_FP_FAST) && !defined(_M_FP_EXCEPT)
	// we are reading past the end of the array and doing arithmetic in the last iteration, but with fp exceptions turned of it's fine 
	for (u32 i = 0; i < batches_num; ++i, A += size_sq, Y += size_sq) {
		for (s32 j = 0; j < size; ++j, B += size) {
			const __m128 b0 = _mm_loadu_ps(B);
			const __m128 b1 = _mm_loadu_ps(B + stride_sse);
			for (s32 k = 0; k < size_sq; k += size) {
				__m128 a0 = _mm_loadu_ps(A + k);
				__m128 a1 = _mm_loadu_ps(A + stride_sse + k);
				a0 = _mm_mul_ps(a0, b0);
				a1 = _mm_mul_ps(a1, b1);
				a1 = _mm_and_ps(a1, *(__m128*) & rem_mask_m128);
				a0 = _mm_add_ps(a0, a1);
				Y[k + j] = horizontal_add(a0);
			}
}
	}
#	else
	const __m128i rem_mask_m128_inv = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0);
	for (u32 i = 0; i < batches_num - 1; ++i, A += size_sq, Y += size_sq) {
		for (s32 j = 0; j < size; ++j, B += size) {
			const __m128 b0 = _mm_loadu_ps(B);
			const __m128 b1 = _mm_loadu_ps(B + stride_sse);
			for (s32 k = 0; k < size_sq; k += size) {
				__m128 a0 = _mm_loadu_ps(A + k);
				__m128 a1 = _mm_loadu_ps(A + stride_sse + k);
				a0 = _mm_mul_ps(a0, b0);
				a1 = _mm_mul_ps(a1, b1);
				a1 = _mm_and_ps(a1, *(__m128*) & rem_mask_m128);
				a0 = _mm_add_ps(a0, a1);
				Y[k + j] = horizontal_add(a0);
			}
		}
	}
	// compute the last matrix in the batch
	for (s32 j = 0; j < size - 1; ++j, B += size) {
		const __m128 b0 = _mm_loadu_ps(B);
		__m128 b1 = _mm_loadu_ps(B + stride_sse);
		for (s32 k = 0; k < size_sq; k += size) {
			__m128 a0 = _mm_loadu_ps(A + k);
			__m128 a1 = _mm_loadu_ps(A + stride_sse + k);
			a0 = _mm_mul_ps(a0, b0);
			a1 = _mm_mul_ps(a1, b1);
			a1 = _mm_and_ps(a1, *(__m128*) & rem_mask_m128);
			a0 = _mm_add_ps(a0, a1);
			Y[k + j] = horizontal_add(a0);
		}
	}
	// last vector in the last matrix
	A -= 2;
	B -= 2;
	const __m128 b0 = _mm_loadu_ps(B);
	__m128 b1 = _mm_loadu_ps(B + stride_sse);
	for (s32 k = 0; k < size_sq; k += size) {
		__m128 a0 = _mm_loadu_ps(A + k);
		__m128 a1 = _mm_loadu_ps(A + stride_sse + k);
		a0 = _mm_mul_ps(a0, b0);
		a0 = _mm_and_ps(a0, *(__m128*) & rem_mask_m128_inv);
		a1 = _mm_mul_ps(a1, b1);
		a0 = _mm_add_ps(a0, a1);
		Y[k + size - 1] = horizontal_add(a0);
	}
#	endif	//  defined(_M_FP_FAST) && !defined(_M_FP_EXCEPT)
#endif
}

void generate_data(fp32* A, fp32* B, u32 size) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> fp_distr(0.0f, 1.0f);

	for (u32 w = 0; w < size; w++) {
		A[w] = fp_distr(gen);
		B[w] = fp_distr(gen);
	}
}

static void mat_mat_mul(benchmark::State& state) {
	constexpr u32 size = 6;
	constexpr u32 size_sq = size * size;
	constexpr u32 batch_num = 26;

	typedef __m256 m_reg;

	fp32* A = (fp32*)malloc(size_sq * batch_num * sizeof(fp32));
	fp32* B = (fp32*)malloc(size_sq * batch_num * sizeof(fp32));
	fp32* Y = (fp32*)malloc(size_sq * batch_num * sizeof(fp32));

	generate_data(A, B, size_sq * batch_num);
    for (auto _ : state) {     
		batch_mat_mul_transposed_6x6(A, B, Y, batch_num);
        benchmark::DoNotOptimize(Y);
    }
	free(A);
	free(B);
	free(Y);
}

BENCHMARK(mat_mat_mul);

static void mat_mat_mul_simd(benchmark::State& state) {
	constexpr u32 size = 6;
	constexpr u32 size_sq = size * size;
	constexpr u32 batch_num = 26;

	typedef __m256 m_reg;

	fp32* A = (fp32*)malloc(size_sq * batch_num * sizeof(fp32));
	fp32* B = (fp32*)malloc(size_sq * batch_num * sizeof(fp32));
	fp32* Y = (fp32*)malloc(size_sq * batch_num * sizeof(fp32));

	generate_data(A, B, size_sq * batch_num);
	for (auto _ : state) {
		batch_mat_mul_transposed_6x6_simd(A, B, Y, batch_num);
		benchmark::DoNotOptimize(Y);
	}
	free(A);
	free(B);
	free(Y);
}

BENCHMARK(mat_mat_mul_simd);

BENCHMARK_MAIN();


//template <typename T>
//bool	fp_similar(T a, T b, T cmp = T(0.00001f)) { return abs(a - b) <= cmp; }
//
//int main() {
//	constexpr u32 size = 6;
//	constexpr u32 size_sq = size * size;
//	constexpr u32 batch_num = 26;
//
//	typedef __m256 m_reg;
//
//	fp32* A = (fp32*)malloc(size_sq * batch_num * sizeof(fp32));
//	fp32* B = (fp32*)malloc(size_sq * batch_num * sizeof(fp32));
//	fp32* Y_0 = (fp32*)malloc(size_sq * batch_num * sizeof(fp32));
//	fp32* Y_1 = (fp32*)malloc(size_sq * batch_num * sizeof(fp32));
//
//	generate_data(A, B, size_sq * batch_num);
//
//	batch_mat_mul_transposed_6x6(A, B, Y_0, batch_num);
//	batch_mat_mul_transposed_6x6_simd(A, B, Y_1, batch_num);
//
//	for (int i = 0; i < size_sq * batch_num; i++) {
//		if (!fp_similar(Y_0[i], Y_1[i], 0.00001f))
//			printf("%.2f, %.2f, %d\n", Y_0[i], Y_1[i], i);
//	}
//
//	free(A);
//	free(B);
//	free(Y_0);
//	free(Y_1);
//
//	return 0;
//}
