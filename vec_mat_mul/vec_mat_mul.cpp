#include <stdint.h>
#include <assert.h>
#include <memory>
#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <random>

#define __AVX__
#ifdef __AVX__
#define USE_HALF_FLOAT
#endif // __AVX__



typedef float fp32;
typedef uint8_t u8;
typedef uint16_t fp16;
typedef int32_t s32;
typedef uint32_t u32;
typedef uint64_t u64;

static const alignas(alignof(__m256i)) u32 rem_mask_table[8][8] = {
	{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
	{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xFFFFFFFF},
	{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF},
	{0x0, 0x0, 0x0, 0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
	{0x0, 0x0, 0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,},
	{0x0, 0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
	{0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,},
	{0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
};

template <typename T>
void vec_mat_mul(fp32* res, const T* tensor, const fp32* vector, u32 height, u32 width) {	// height == row, width = col
	constexpr bool is_half_float = (sizeof(T) == sizeof(fp16));
#ifdef __AVX__
	static_assert(sizeof(T) == sizeof(fp16) || sizeof(T) == sizeof(fp32), "Unsupported type");
	assert(0 == ((u64)vector & (sizeof(__m256) - 1)));
	assert(0 == ((u64)res & (sizeof(__m256) - 1)));
	constexpr u32 step_sz_256_fp32 = sizeof(__m256) / sizeof(fp32);
	const u32 rem = height & (step_sz_256_fp32 - 1); // [0:7] -- works for both fp16 and fp32 == (height & (step_128_sz_fp16 - 1))
	const u32 height_trunc = height - rem;
	const u32 rem_offset = height_trunc - (step_sz_256_fp32 - rem);
	if (is_half_float) { // compile time branch
		constexpr u32 step_128_sz_fp16 = sizeof(__m128i) / sizeof(fp16);
		const __m256i rem_mask = _mm256_load_si256((__m256i*)rem_mask_table[rem]);
		__m256 rem_vec = _mm256_maskload_ps(&vector[rem_offset], rem_mask);
		fp16* t_row = (fp16*)tensor;
		for (u32 w = 0; w < width; w++) {
			fp32* out	= &res[w];
			__m256 accum_256 = _mm256_setzero_ps();
			const fp32 *vs = vector;
			for (u32 h = 0; h < height_trunc; h += step_128_sz_fp16) {
				const __m256 &v = *(__m256*)vs;
				__m128i t_ph = _mm_loadu_si128((__m128i*)(&t_row[h]));
				__m256 t_ps = _mm256_cvtph_ps(t_ph);
				t_ps = _mm256_mul_ps(t_ps, v);
				accum_256 = _mm256_add_ps(accum_256, t_ps);
				vs += step_sz_256_fp32;
			}
			// compute reminder chunk
			__m128i t_ph = _mm_loadu_si128((__m128i*)(&t_row[rem_offset]));
			__m256 t_ps = _mm256_cvtph_ps(t_ph);
			t_ps = _mm256_mul_ps(t_ps, rem_vec);
			accum_256 = _mm256_add_ps(accum_256, t_ps);
			// horizontal add
			__m128 hi = _mm256_extractf128_ps(accum_256, 1);
			__m128 lo = _mm256_extractf128_ps(accum_256, 0);
			__m128 accum = _mm_add_ps(hi, lo);
			__m128 shuf = _mm_shuffle_ps(accum, accum, _MM_SHUFFLE(2, 3, 0, 1));
			__m128 sums = _mm_add_ps(accum, shuf);
			shuf = _mm_movehl_ps(shuf, sums);
			sums = _mm_add_ss(sums, shuf);
			*out = _mm_cvtss_f32(sums);
			t_row += height;
		}
	} else {
		__m256i rem_mask = _mm256_load_si256((__m256i*)rem_mask_table[rem]);
		__m256 rem_vec = _mm256_maskload_ps(&vector[rem_offset], rem_mask);
		fp32* t_row = (fp32*)tensor;
		for (u32 w = 0; w < width; w++) {
				fp32* out = &res[w];
			__m256 accum256 = _mm256_setzero_ps();
			const fp32* vs = vector;
			for (u32 h = 0; h < height_trunc; h += step_sz_256_fp32) {
				const __m256& v = *(__m256*)vs;
				__m256 t_ps = _mm256_loadu_ps(&t_row[h]);
				t_ps = _mm256_mul_ps(t_ps, v);
				accum256 = _mm256_add_ps(accum256, t_ps);
				vs += step_sz_256_fp32;
			}
			// compute reminder chunk
			__m256 t_ps = _mm256_loadu_ps(&t_row[rem_offset]);
			t_ps = _mm256_mul_ps(t_ps, rem_vec);
			accum256 = _mm256_add_ps(accum256, t_ps);
			// horizontal add
			__m128 hi = _mm256_extractf128_ps(accum256, 1);
			__m128 lo = _mm256_extractf128_ps(accum256, 0);
			__m128 accum = _mm_add_ps(hi, lo);
			__m128 shuf = _mm_shuffle_ps(accum, accum, _MM_SHUFFLE(2, 3, 0, 1));
			__m128 sums = _mm_add_ps(accum, shuf);
			shuf = _mm_movehl_ps(shuf, sums);
			sums = _mm_add_ss(sums, shuf);
			*out = _mm_cvtss_f32(sums);
			t_row += height;
		}
	}
#else // __SSE2__ version
	static_assert(sizeof(T) == sizeof(fp32), "fp16 not supported without AVX/f16c");
	assert(0 == ((u64)vector & (sizeof(__m128) - 1)));
	assert(0 == ((u64)res & (sizeof(__m128) - 1)));
	constexpr u32 step_sz_128_fp32 = sizeof(__m128) / sizeof(fp32);
	const u32 rem = height & (step_sz_128_fp32 - 1); // [0:3]
	const __m128i rem_mask = _mm_load_si128((__m128i*)&rem_mask_table[rem][step_sz_128_fp32]);
	const u32 height_trunc = height - rem;
	const u32 rem_offset = height_trunc - (step_sz_128_fp32 - rem);
	__m128 rem_vec = _mm_loadu_ps(&vector[rem_offset]);
	rem_vec = _mm_and_ps(rem_vec, *(__m128*)&rem_mask);
	fp32* t_row = (fp32*)tensor;
	for (u32 w = 0; w < width; w++) {
		fp32* out = (fp32*)&res[w];
		__m128 accum = _mm_setzero_ps();
		const fp32* vs = vector;
		for (u32 h = 0; h < height_trunc; h += step_sz_128_fp32) {
			const __m128& v = *(__m128*)vs;
			__m128 t_ps = _mm_loadu_ps(&t_row[h]);
			t_ps = _mm_mul_ps(t_ps, v);
			accum = _mm_add_ps(accum, t_ps);
			vs += step_sz_128_fp32;
		}
		// compute reminder chunk
		__m128 t_ps = _mm_loadu_ps(&t_row[rem_offset]);
		t_ps = _mm_mul_ps(t_ps, rem_vec);
		accum = _mm_add_ps(accum, t_ps);
		// horizontal add
		__m128 shuf = _mm_shuffle_ps(accum, accum, _MM_SHUFFLE(2, 3, 0, 1));
		__m128 sums = _mm_add_ps(accum, shuf);
		shuf = _mm_movehl_ps(shuf, sums);
		sums = _mm_add_ss(sums, shuf);
		*out = _mm_cvtss_f32(sums);
		t_row += height;
	}
#endif
}

template <typename T>
void simple_vec_mat_mul(fp32* res, const T* tensor, fp32* sub_tensor, const fp32* vector, u32 height, u32 width) {
	static_assert(sizeof(T) == sizeof(fp16) || sizeof(T) == sizeof(fp32), "Unsupported type");
	constexpr bool is_half_float = (sizeof(T) == sizeof(fp16));
	memset(res, 0, sizeof(fp32) * width);
	if (is_half_float) { // compile time branch
		alignas(alignof(__m128)) fp32 ps[4];
		for (u32 w = 0; w < width; w++) {
			for (u32 h = 0; h < height; h++) {
				__m128i t_ph = _mm_loadu_epi16((void*)(&tensor[w * height + h]));
				*(__m128*)ps = _mm_cvtph_ps(t_ph);
				sub_tensor[w * height + h] += ps[0];
			}
		}
	} else {		
		for (u32 w = 0; w < width; w++) {
			for (u32 h = 0; h < height; h++) {
				sub_tensor[w * height + h] += tensor[w * height + h];
			}
		}
	}
	for (u32 w = 0; w < width; w++) {
		for (u32 h = 0; h < height; h++) {
			res[w] += sub_tensor[w * height + h] * vector[h];
		}
	}
}

template <typename T>
void generate_data(T* tensor, fp32* vector, u32 height, u32 width) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> fp_distr(0.0f, 1.0f);

	constexpr bool is_half_float = (sizeof(T) == sizeof(fp16));
	// fill vector
	for (u32 h = 0; h < height; h++) {
		vector[h] = fp_distr(gen);
	}
	alignas(alignof(__m128)) float ps[sizeof(__m128) / sizeof(fp32)];
	// fill tensor
	for (u32 w = 0; w < width; w++) {
		T* t_row = (T*)&tensor[w * height];
		for (u32 h = 0; h < height; h++) {
			ps[0] = fp_distr(gen);
			ps[1] = fp_distr(gen);
			ps[2] = fp_distr(gen);
			ps[3] = fp_distr(gen);
			if (is_half_float) {
				__m128i ph = _mm_cvtps_ph(*(__m128*)ps, 0);
				_mm_storeu_epi64((void*)&t_row[h], ph);
			} else {
				_mm_storeu_ps((fp32*)&t_row[h], *(__m128*)ps);
			}
		}
	}
}

template <typename T>
bool	fp_similar(T a, T b, T cmp = T(0.00001f)) { return abs(a - b) <= cmp; }

int main() {
	constexpr u32 h = 510;
	constexpr u32 w = 288;

#ifdef USE_HALF_FLOAT
	typedef fp16 f_type;
#else	// USE_HALF_FLOAT
	typedef fp32 f_type;
#endif	// USE_HALF_FLOAT

#ifdef __AVX__
	typedef __m256 m_reg;
#else // __AVX2__
	typedef __m128 m_reg;
#endif // __AVX2__

	alignas(alignof(m_reg)) fp32 vector[h];
	f_type*tensor = (f_type*)malloc(w * h * sizeof(f_type) + sizeof(m_reg));

	generate_data(tensor, vector, h, w);

	alignas(alignof(m_reg)) fp32 result_a[w];
	vec_mat_mul(result_a, tensor, vector, h, w);

	const u32 sub_tensor_sz_bytes = w * h * sizeof(fp32) + sizeof(m_reg);
	fp32* sub_tensor = (fp32*)malloc(sub_tensor_sz_bytes);
	memset(sub_tensor, 0, sub_tensor_sz_bytes);
	alignas(alignof(m_reg)) fp32 result_b[w];
	simple_vec_mat_mul(result_b, tensor, sub_tensor, vector, h, w);

	for (u32 i = 0; i < w; i++) {
		if (!fp_similar(result_a[i], result_b[i], 0.001f)) {
			printf("result_a[%d]: %.4f != result_b[%d]: %.4f\n", i, result_a[i], i, result_b[i]);
		}
	}

	free(tensor);
	free(sub_tensor);
	return 0;
}
