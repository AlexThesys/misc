#include <stdint.h>
#include <assert.h>
#include <memory>
#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <random>
#include <string.h>

#define USE_HALF_FLOAT

typedef float fp32;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint16_t fp16;
typedef int32_t s32;
typedef uint32_t u32;
typedef uint64_t u64;

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

inline fp32 horizontal_add(const __m128& accum) {
	__m128 shuf = _mm_shuffle_ps(accum, accum, _MM_SHUFFLE(2, 3, 0, 1));
	__m128 sums = _mm_add_ps(accum, shuf);
	shuf = _mm_movehl_ps(shuf, sums);
	sums = _mm_add_ss(sums, shuf);
	return _mm_cvtss_f32(sums);
}

fp32 half_to_float(fp16 h) {
	const u32 e = (h & 0x7C00);
	const u32 f = ((h & 0x8000) << 16) | ((e == 0) - 1) & (((e + 0x1C000) | (h & 0x03FF)) << 13);
	return *(fp32*)&f;
}

inline __m128 cvtfp16_fp32(const void* ph) {
	const __m128i zero = _mm_setzero_si128();
	const __m128i s_mask = _mm_set1_epi32(0x00008000);
	const __m128i e_mask = _mm_set1_epi32(0x00007C00);
	const __m128i m_mask = _mm_set1_epi32(0x000003FF);
	const __m128i e_bias = _mm_set1_epi32(0x0001C000);
	__m128i x = _mm_loadu_si64(ph);
	x = _mm_unpacklo_epi16(x, zero);
	__m128i s = _mm_and_si128(x, s_mask);
	__m128i e = _mm_and_si128(x, e_mask);
	__m128i m = _mm_and_si128(x, m_mask);
	const __m128i zero_mask = _mm_cmpgt_epi32(e, zero);	// it won't be less than zero
	s = _mm_slli_epi32(s, 16);
	e = _mm_add_epi32(e, e_bias);
	x = _mm_or_si128(m, e);
	x = _mm_slli_epi32(x, 13);
	x = _mm_and_si128(x, zero_mask);
	x = _mm_or_si128(x, s);
	return *(__m128*)&x;
}

template <typename T>
void vec_mat_mul(fp32* res, const T* tensor, const fp32* vector, u32 height, u32 width) {	// height == row, width = col
	constexpr bool is_half_float = (sizeof(T) == sizeof(fp16));
#ifdef __AVX__
	static_assert(sizeof(T) == sizeof(fp16) || sizeof(T) == sizeof(fp32), "Unsupported type");
	assert(0 == ((u64)vector & (sizeof(__m256) - 1)));
	assert(0 == ((u64)res & (sizeof(__m256) - 1)));
	const u32 rem = height & (stride_avx - 1);
	const u32 height_trunc = height - rem;
	const u32 rem_offset = height_trunc - (stride_avx - rem);
	if (is_half_float) { // compile time branch
		const __m256i rem_mask = _mm256_load_si256((__m256i*)rem_mask_table[rem]);
		__m256 rem_vec = _mm256_maskload_ps(&vector[rem_offset], rem_mask);
		const fp16* t_row = (fp16*)tensor;
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
			__m128 accum = _mm_add_ps(hi, *(__m128*)&accum_256);
			*out = horizontal_add(accum);
		}
	}
	else {
		__m256i rem_mask = _mm256_load_si256((__m256i*)rem_mask_table[rem]);
		__m256 rem_vec = _mm256_maskload_ps(&vector[rem_offset], rem_mask);
		const fp32* t_row = (fp32*)tensor;
		for (u32 w = 0; w < width; w++, t_row += height) {
			fp32* out = &res[w];
			__m256 accum_256 = _mm256_setzero_ps();
			for (u32 h = 0; h < height_trunc; h += stride_avx) {
				__m256 t_ps = _mm256_loadu_ps(&t_row[h]);
				t_ps = _mm256_mul_ps(t_ps, *(__m256*)(vector + h));
				accum_256 = _mm256_add_ps(accum_256, t_ps);
			}
			// compute reminder chunk
			__m256 t_ps = _mm256_loadu_ps(&t_row[rem_offset]);
			t_ps = _mm256_mul_ps(t_ps, rem_vec);
			accum_256 = _mm256_add_ps(accum_256, t_ps);
			// horizontal add
			__m128 hi = _mm256_extractf128_ps(accum_256, 1);
			__m128 accum = _mm_add_ps(hi, *(__m128*)&accum_256);
			*out = horizontal_add(accum);
		}
	}
#else // __SSE2__ version
	if (is_half_float) { // compile time branch
		assert(0 == ((u64)vector & (sizeof(__m128) - 1)));
		assert(0 == ((u64)res & (sizeof(__m128) - 1)));
		const u32 rem = height & (stride_sse - 1);
		const __m128i rem_mask = _mm_load_si128((__m128i*) & rem_mask_table[rem][stride_sse]);
		const u32 height_trunc = height - rem;
		const u32 rem_offset = height_trunc - (stride_sse - rem);
		__m128 rem_vec = _mm_loadu_ps(&vector[rem_offset]);
		rem_vec = _mm_and_ps(rem_vec, *(__m128*) & rem_mask);
		const fp16* t_row = (fp16*)tensor;
		for (u32 w = 0; w < width; w++, t_row += height) {
			fp32* out = (fp32*)&res[w];
			__m128 accum = _mm_setzero_ps();
			for (u32 h = 0; h < height_trunc; h += stride_sse) {
				__m128 t_ps = cvtfp16_fp32(&t_row[h]);
				t_ps = _mm_mul_ps(t_ps, *(__m128*)(vector + h));
				accum = _mm_add_ps(accum, t_ps);
			}
			// compute reminder chunk
			__m128 t_ps = cvtfp16_fp32(&t_row[rem_offset]);
			t_ps = _mm_mul_ps(t_ps, rem_vec);
			accum = _mm_add_ps(accum, t_ps);
			// horizontal add
			*out = horizontal_add(accum);
		}
	} else {
		assert(0 == ((u64)vector & (sizeof(__m128) - 1)));
		assert(0 == ((u64)res & (sizeof(__m128) - 1)));
		const u32 rem = height & (stride_sse - 1);
		const __m128i rem_mask = _mm_load_si128((__m128i*) & rem_mask_table[rem][stride_sse]);
		const u32 height_trunc = height - rem;
		const u32 rem_offset = height_trunc - (stride_sse - rem);
		__m128 rem_vec = _mm_loadu_ps(&vector[rem_offset]);
		rem_vec = _mm_and_ps(rem_vec, *(__m128*) & rem_mask);
		const fp32* t_row = (fp32*)tensor;
		for (u32 w = 0; w < width; w++, t_row += height) {
			fp32* out = (fp32*)&res[w];
			__m128 accum = _mm_setzero_ps();
			for (u32 h = 0; h < height_trunc; h += stride_sse) {
				__m128 t_ps = _mm_loadu_ps(&t_row[h]);
				t_ps = _mm_mul_ps(t_ps, *(__m128*)(vector + h));
				accum = _mm_add_ps(accum, t_ps);
			}
			// compute reminder chunk
			__m128 t_ps = _mm_loadu_ps(&t_row[rem_offset]);
			t_ps = _mm_mul_ps(t_ps, rem_vec);
			accum = _mm_add_ps(accum, t_ps);
			// horizontal add
			*out = horizontal_add(accum);
		}
	}
#endif
}

template <typename T>
void simple_vec_mat_mul(fp32* res, const T* tensor, const fp32* vector, u32 height, u32 width) {
	static_assert(sizeof(T) == sizeof(fp16) || sizeof(T) == sizeof(fp32), "Unsupported type");
	constexpr bool is_half_float = (sizeof(T) == sizeof(fp16));
	for (u32 w = 0; w < width; w++) {
		for (u32 h = 0; h < height; h++) {
			if (is_half_float) { // compile time branch
				const fp32 f = half_to_float(tensor[w * height + h]);
				res[w] += f * vector[h];
			}
			else {
				res[w] += tensor[w * height + h] * vector[h];
			}
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
			}
			else {
				_mm_storeu_ps((fp32*)&t_row[h], *(__m128*)ps);
			}
		}
	}
}

template <typename T>
bool	fp_similar(T a, T b, T cmp = T(0.00001f)) { return abs(a - b) <= cmp; }

int main() {

	constexpr u32 height = 1000;
	constexpr u32 width = 700;
	constexpr u32 max_size = 8;

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
	alignas(alignof(m_reg)) fp32 vector[height+max_size];
	alignas(alignof(m_reg)) fp32 result_a[width+max_size];
	alignas(alignof(m_reg)) fp32 result_b[width + max_size];
	f_type* tensor = (f_type*)malloc((width+max_size) * (height + max_size) * sizeof(f_type));
	for (u32 h = height, w = width, sz = height + max_size; h < sz; h++, w++) {

		generate_data(tensor, vector, h, w);


		vec_mat_mul(result_a, tensor, vector, h, w);

		memset(result_b, 0, w * sizeof(fp32));
		simple_vec_mat_mul(result_b, tensor, vector, h, w);

		for (u32 i = 0; i < w; i++) {
			//assert(fp_similar(result_a[i], result_b[i], 0.001f));
			if (!fp_similar(result_a[i], result_b[i], 0.001f)) {
				printf("result_a[%d]: %.4f != result_b[%d]: %.4f\n", i, result_a[i], i, result_b[i]);
			}
		}
	}
	free(tensor);


	return 0;
}
