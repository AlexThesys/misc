#include <stdint.h>
#include <assert.h>
#include <memory>
#include <math.h>
#include <stdio.h>

#define __AVX2__
#define USE_MASKLOAD
//#define USE_RSHIFT
#define USE_HALF_FLOAT

#include <immintrin.h>

#include <random>

typedef float fp32;
typedef uint16_t fp16;
typedef int32_t s32;
typedef uint32_t u32;

#define VECT_MAX_SIZE 1024


static const alignas(alignof(__m128i)) u32 rem_mask_128[8][4] = { 
	{0x0, 0x0, 0x0, 0x0},
	{0x0000FFFF, 0x0, 0x0, 0x0},
	{0xFFFFFFFF, 0x0, 0x0, 0x0},
	{0xFFFFFFFF, 0x0000FFFF, 0x0, 0x0},
	{0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0},
	{0xFFFFFFFF, 0xFFFFFFFF, 0x0000FFFF, 0x0},
	{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0},
	{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0000FFFF},
};

#ifdef __AVX2__
static const alignas(alignof(__m256i)) u32 rem_mask_256[8][8] = {
	{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
	{0xFFFFFFFF, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
	{0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
	{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0, 0x0, 0x0, 0x0},
	{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0, 0x0, 0x0},
	{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0, 0x0},
	{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0},
	{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0},
};
#endif  // __AVX2__

/*vector has to be padded with zeros to be a multuple of xmm/mm64 register size*/
/*total number of elements in tensor has to be padded with zeros to be a multuple of xmm/mm64 register size + 1*/
/*res should be zero initiallized*/
template <typename T>
void vec_mat_mul(fp32* res, const T* tensor, const T* vector, const fp32 *scale, u32 height, u32 width) {	// height == row, width = col
	static_assert(sizeof(T) == sizeof(fp16) || sizeof(T) == sizeof(fp32), "Unsupported type");
	constexpr u32 dim = 4;
	constexpr bool is_half_float = (sizeof(T) == sizeof(fp16));
	constexpr u32 bits_in_byte = 8;
#ifdef __AVX2__
	constexpr u32 step_sz_256_fp32 = sizeof(__m256) / sizeof(fp32);
	if (is_half_float) { // compile time branch
		constexpr u32 step_128_sz_fp16 = sizeof(__m128i) / sizeof(fp16);
		const u32 rem = height & (step_128_sz_fp16 - 1);	// [0:7]
		__m128i rem_mask = _mm_load_si128((__m128i*)rem_mask_128[rem]);
		const u32 height_trunc = height - rem;
		for (u32 d = 0; d < dim; d++) {
			alignas(alignof(__m256)) fp32 vec_scaled[VECT_MAX_SIZE];	// can be h_uvector in out case
			fp32* vs = vec_scaled;
			__m256 s = _mm256_set1_ps(scale[d]);
			for (u32 i = 0; i < height; i += step_128_sz_fp16) {	// this is going to work for the vector, because it's padded with zeroes
				__m128i v_ph = _mm_loadu_si128((__m128i*)(&vector[i]));
				__m256 v_ps = _mm256_cvtph_ps(v_ph);
				*(__m256*)(vs) = _mm256_mul_ps(v_ps, s);
				vs += step_sz_256_fp32;
			}
			fp16* t_mat = (fp16*)&tensor[d * width * height];
			for (u32 w = 0; w < width; w++) {
				fp16* t_row = &t_mat[w * height];
				fp32* out	= &res[w];
				__m256 accum_256 = _mm256_setzero_ps();
				const fp32 *vs = vec_scaled;
				for (u32 h = 0; h < height_trunc; h += step_128_sz_fp16) {
					const __m256 &v = *(__m256*)vs;
					__m128i t_ph = _mm_loadu_si128((__m128i*)(&t_row[h]));
					__m256 t_ps = _mm256_cvtph_ps(t_ph);
					t_ps = _mm256_mul_ps(t_ps, v);
					accum_256 = _mm256_add_ps(accum_256, t_ps);
					vs += step_sz_256_fp32;
				}
				// compute reminder chunk
                //if (rem != 0) {
                    const __m256& v = *(__m256*)vs;
                    __m128i t_ph = _mm_loadu_si128((__m128i*)(&t_row[height_trunc]));
                    t_ph = _mm_and_si128(t_ph, rem_mask);
                    __m256 t_ps = _mm256_cvtph_ps(t_ph);
                    t_ps = _mm256_mul_ps(t_ps, v);
                    accum_256 = _mm256_add_ps(accum_256, t_ps);
                //}
                // horizontal add
				__m128 hi = _mm256_extractf128_ps(accum_256, 1);
				__m128 lo = _mm256_extractf128_ps(accum_256, 0);
				__m128 accum = _mm_add_ps(hi, lo);
				__m128 shuf = _mm_shuffle_ps(accum, accum, _MM_SHUFFLE(2, 3, 0, 1));
				__m128 sums = _mm_add_ps(accum, shuf);
				shuf = _mm_movehl_ps(shuf, sums);
				sums = _mm_add_ss(sums, shuf);
				*out += _mm_cvtss_f32(sums);
			}
		}
	} else {
		const u32 rem = height & (step_sz_256_fp32 - 1); // [0:7]
		__m256i rem_mask = _mm256_load_si256((__m256i*)rem_mask_256[rem]);

		const u32 height_trunc = height - rem;
		for (u32 d = 0; d < dim; d++) {
			alignas(alignof(__m256)) fp32 vec_scaled[VECT_MAX_SIZE];
			fp32* vs = vec_scaled;
			__m256 s = _mm256_set1_ps(scale[d]);
			for (u32 i = 0; i < height; i += step_sz_256_fp32) {	// this is going to work for the vector, because it's padded with zeroes
				__m256 v_ps = _mm256_loadu_ps((fp32*)&vector[i]);
				*(__m256*)(vs) = _mm256_mul_ps(v_ps, s);
				vs += step_sz_256_fp32;
			}
			fp32* t_mat = (fp32*)&tensor[d * width * height];
			for (u32 w = 0; w < width; w++) {
				fp32* t_row = &t_mat[w * height];
				fp32* out = &res[w];
				__m256 accum256 = _mm256_setzero_ps();
				const fp32* vs = vec_scaled;
				for (u32 h = 0; h < height_trunc; h += step_sz_256_fp32) {
					const __m256& v = *(__m256*)vs;
					__m256 t_ps = _mm256_loadu_ps(&t_row[h]);
					t_ps = _mm256_mul_ps(t_ps, v);
					accum256 = _mm256_add_ps(accum256, t_ps);
					vs += step_sz_256_fp32;
				}
				// compute reminder chunk
                //if (rem != 0) {
                    const __m256& v = *(__m256*)vs;
#ifndef USE_MASKLOAD
                    __m256i t_si = _mm256_loadu_si256((__m256i*)(&t_row[height_trunc]));
                    t_si = _mm256_and_si256(t_si, rem_mask);
                    __m256 t_ps = *(__m256*)&t_si;
#else // USE_MASKLOAD
                    __m256 t_ps = _mm256_maskload_ps(&t_row[height_trunc], rem_mask);
#endif // USE_MASKLOAD
                    t_ps = _mm256_mul_ps(t_ps, v);
                    accum256 = _mm256_add_ps(accum256, t_ps);
                //}
				// horizontal add
				__m128 hi = _mm256_extractf128_ps(accum256, 1);
				__m128 lo = _mm256_extractf128_ps(accum256, 0);
				__m128 accum = _mm_add_ps(hi, lo);
				__m128 shuf = _mm_shuffle_ps(accum, accum, _MM_SHUFFLE(2, 3, 0, 1));
				__m128 sums = _mm_add_ps(accum, shuf);
				shuf = _mm_movehl_ps(shuf, sums);
				sums = _mm_add_ss(sums, shuf);
				*out += _mm_cvtss_f32(sums);
			}
		}
	}
#else // __SSE2__ version
	constexpr u32 step_sz_128_fp32 = sizeof(__m128) / sizeof(fp32);
	if (is_half_float) { // compile time branch
#ifndef USE_RSHIFT
		constexpr u32 step_64_sz_fp16 = sizeof(__m64) / sizeof(fp16);
		const u32 rem = height & (step_64_sz_fp16 - 1);	// [0:3]
		__m128i rem_mask = _mm_load_si128((__m128i*)rem_mask_128[rem]);
		const u32 height_trunc = height - rem;
		for (u32 d = 0; d < dim; d++) {
			alignas(alignof(__m128)) fp32 vec_scaled[VECT_MAX_SIZE];
			fp32* vs = vec_scaled;
			__m128 s = _mm_set1_ps(scale[d]);
			for (u32 i = 0; i < height; i += step_64_sz_fp16) {	// this is going to work for the vector, because it's padded with zeroes
				__m128i v_ph = _mm_loadu_si64((void*)(&vector[i]));
				__m128 v_ps = _mm_cvtph_ps(v_ph);
				*(__m128*)(vs) = _mm_mul_ps(v_ps, s);
				vs += step_sz_128_fp32;
			}
			fp16* t_mat = (fp16*)&tensor[d * width * height];
			for (u32 w = 0; w < width; w++) {
				fp16* t_row = &t_mat[w * height];
				fp32* out = &res[w];
				__m128 accum = _mm_setzero_ps();
				const fp32* vs = vec_scaled;
				for (u32 h = 0; h < height_trunc; h += step_64_sz_fp16) {
					const __m128& v = *(__m128*)vs;
					__m128i t_ph = _mm_loadu_si64((void*)(&t_row[h]));
					__m128 t_ps = _mm_cvtph_ps(t_ph);
					t_ps = _mm_mul_ps(t_ps, v);
					accum = _mm_add_ps(accum, t_ps);
					vs += step_sz_128_fp32;
				}
				// compute reminder chunk
                //if (rem != 0) {
                    const __m128& v = *(__m128*)vs;
                    __m128i t_ph = _mm_loadu_si128((__m128i*)(&t_row[height_trunc]));
                    t_ph = _mm_and_si128(t_ph, rem_mask);
                    __m128 t_ps = _mm_cvtph_ps(t_ph);
                    t_ps = _mm_mul_ps(t_ps, v);
                    accum = _mm_add_ps(accum, t_ps);
                //}
				// horizontal add
				__m128 shuf = _mm_shuffle_ps(accum, accum, _MM_SHUFFLE(2, 3, 0, 1));
				__m128 sums = _mm_add_ps(accum, shuf);
				shuf = _mm_movehl_ps(shuf, sums);
				sums = _mm_add_ss(sums, shuf);
				*out += _mm_cvtss_f32(sums);
			}
		}
#else // USE_RSHIFT
		constexpr u32 step_128_sz_fp16 = sizeof(__m128i) / sizeof(fp16);
		constexpr u32 step_64_sz_fp16 = sizeof(__m64) / sizeof(fp16);
		const u32 rem = height & (step_128_sz_fp16 - 1);	// [0:7]
		__m128i rem_mask = _mm_load_si128((__m128i*)rem_mask_128[rem]);
		const u32 height_trunc = height - rem;
		for (u32 d = 0; d < dim; d++) {
			alignas(alignof(__m128)) fp32 vec_scaled[VECT_MAX_SIZE];
			fp32* vs = vec_scaled;
			__m128 s = _mm_set1_ps(scale[d]);
			for (u32 i = 0; i < height; i += step_128_sz_fp16) {	// this is going to work for the vector, because it's padded with zeroes
				__m128i v_ph = _mm_loadu_si128((__m128i*)(&vector[i]));
				__m128 v_ps = _mm_cvtph_ps(v_ph);
				*(__m128*)(vs) = _mm_mul_ps(v_ps, s);
				vs += step_64_sz_fp16;
				v_ph = _mm_srli_si128(v_ph, sizeof(__m64));
				v_ps = _mm_cvtph_ps(v_ph);
				*(__m128*)(vs) = _mm_mul_ps(v_ps, s);
				vs += step_64_sz_fp16;
			}
			fp16* t_mat = (fp16*)&tensor[d * width * height];
			for (u32 w = 0; w < width; w++) {
				fp16* t_row = &t_mat[w * height];
				fp32* out = (fp32*)&res[w];
				__m128 accum = _mm_setzero_ps();
				const fp32* vs = vec_scaled;
				for (u32 h = 0; h < height_trunc; h += step_128_sz_fp16) {
					const __m128* v = (__m128*)vs;
					__m128i t_ph = _mm_loadu_si128((__m128i*)(&t_row[h]));
					__m128 t_ps = _mm_cvtph_ps(t_ph);
					t_ps = _mm_mul_ps(t_ps, *v);
					accum = _mm_add_ps(accum, t_ps);
					v = (__m128*)(vs + step_64_sz_fp16);
					t_ph = _mm_srli_si128(t_ph, sizeof(__m64));
					t_ps = _mm_cvtph_ps(t_ph);
					t_ps = _mm_mul_ps(t_ps, *v);
					accum = _mm_add_ps(accum, t_ps);
					vs += step_128_sz_fp16;
				}
				// compute reminder chunk
                //if (rem != 0) {
                    const __m128 *v = (__m128*)vs;
                    __m128i t_ph = _mm_loadu_si128((__m128i*)(&t_row[height_trunc]));
                    t_ph = _mm_and_si128(t_ph, rem_mask);
                    __m128 t_ps = _mm_cvtph_ps(t_ph);
                    t_ps = _mm_mul_ps(t_ps, *v);
                    accum = _mm_add_ps(accum, t_ps);
                    v = (__m128*)(vs + step_64_sz_fp16);
                    t_ph = _mm_srli_si128(t_ph, sizeof(__m64));
                    t_ps = _mm_cvtph_ps(t_ph);
                    t_ps = _mm_mul_ps(t_ps, *v);
                    accum = _mm_add_ps(accum, t_ps);
                //}
				// horizontal add
				__m128 shuf = _mm_shuffle_ps(accum, accum, _MM_SHUFFLE(2, 3, 0, 1));
				__m128 sums = _mm_add_ps(accum, shuf);
				shuf = _mm_movehl_ps(shuf, sums);
				sums = _mm_add_ss(sums, shuf);
				*out += _mm_cvtss_f32(sums);
			}
		}

#endif // USE_RSHIFT
	} else {
		const u32 rem = height & (step_sz_128_fp32 - 1);
		__m128i rem_mask = _mm_load_si128((__m128i*)rem_mask_128[rem+rem]);

		const u32 height_trunc = height - rem;
		for (u32 d = 0; d < dim; d++) {
			alignas(alignof(__m128)) fp32 vec_scaled[VECT_MAX_SIZE];
			fp32* vs = vec_scaled;
			__m128 s = _mm_set1_ps(scale[d]);
			for (u32 i = 0; i < height; i += step_sz_128_fp32) {	// this is going to work for the vector, because it's padded with zeroes
				__m128 v_ps = _mm_loadu_ps((fp32*)&vector[i]);
				*(__m128*)(vs) = _mm_mul_ps(v_ps, s);
				vs += step_sz_128_fp32;
			}
			fp32* t_mat = (fp32*)&tensor[d * width * height];
			for (u32 w = 0; w < width; w++) {
				fp32* t_row = &t_mat[w * height];
				fp32* out = (fp32*)&res[w];
				__m128 accum = _mm_setzero_ps();
				const fp32* vs = vec_scaled;
				for (u32 h = 0; h < height_trunc; h += step_sz_128_fp32) {
					const __m128& v = *(__m128*)vs;
					__m128 t_ps = _mm_loadu_ps(&t_row[h]);
					t_ps = _mm_mul_ps(t_ps, v);
					accum = _mm_add_ps(accum, t_ps);
					vs += step_sz_128_fp32;
				}
				// compute reminder chunk
                //if (rem != 0) {
                    const __m128& v = *(__m128*)vs;
#ifndef USE_MASKLOAD
                    __m128i t_si = _mm_loadu_si128((__m128i*)(&t_row[height_trunc]));
                    t_si = _mm_and_si128(t_si, rem_mask);
                    __m128 t_ps = *(__m128*)&t_si;
#else	// USE_MASKLOAD
                    __m128 t_ps = _mm_maskload_ps(&t_row[height_trunc], rem_mask);
#endif	// USE_MASKLOAD
                    t_ps = _mm_mul_ps(t_ps, v);
                    accum = _mm_add_ps(accum, t_ps);
                //}
				// horizontal add
				__m128 shuf = _mm_shuffle_ps(accum, accum, _MM_SHUFFLE(2, 3, 0, 1));
				__m128 sums = _mm_add_ps(accum, shuf);
				shuf = _mm_movehl_ps(shuf, sums);
				sums = _mm_add_ss(sums, shuf);
				*out += _mm_cvtss_f32(sums);
			}
		}
	}
#endif
}

// v1
//inline fp32 hadd_sse(const __m128 &accum) {
//	__m128 lo= _mm_unpacklo_ps(accum, accum);
//	__m128 hi= _mm_unpackhi_ps(accum, accum);
//	lo = _mm_add_ps(lo, hi);
//	hi = _mm_unpackhi_ps(hi, hi);
//	lo = _mm_add_ss(lo, hi);
//	return	_mm_cvtss_f32(lo);
//}
// v2
//inline fp32	hadd_sse(const __m128 &accum) {
//	__m128 shuf = _mm_shuffle_ps(accum, accum, _MM_SHUFFLE(2, 3, 0, 1));
//	__m128 sums = _mm_add_ps(accum, shuf);
//	shuf = _mm_movehl_ps(shuf, sums);
//	sums = _mm_add_ss(sums, shuf);
//	return _mm_cvtss_f32(sums);
//}

template <typename T>
void simple_vec_mat_mul(fp32* res, const T* tensor, fp32* sub_tensor, const T* vector, const fp32* scale, u32 height, u32 width) {
	static_assert(sizeof(T) == sizeof(fp16) || sizeof(T) == sizeof(fp32), "Unsupported type");
	constexpr u32 dim = 4;
	constexpr bool is_half_float = (sizeof(T) == sizeof(fp16));
	if (is_half_float) { // compile time branch
		alignas(alignof(__m128)) fp32 ps[4];
		for (u32 d = 0; d < dim; d++) {
			for (u32 w = 0; w < width; w++) {
				for (u32 h = 0; h < height; h++) {
					__m128i t_ph = _mm_loadu_epi16((void*)(&tensor[d * width * height + w * height + h]));
					*(__m128*)ps = _mm_cvtph_ps(t_ph);
					sub_tensor[w * height + h] += ps[0] * scale[d];
				}
			}
		}
		for (u32 w = 0; w < width; w++) {
			for (u32 h = 0; h < height; h++) {
				__m128i v_ph = _mm_loadu_epi16((void*)(&vector[h]));
				*(__m128*)ps = _mm_cvtph_ps(v_ph);
				res[w] += sub_tensor[w * height + h] * ps[0];
			}
		}
	} else {		
		for (u32 d = 0; d < dim; d++) {
			for (u32 w = 0; w < width; w++) {
				for (u32 h = 0; h < height; h++) {
					sub_tensor[w * height + h] += tensor[d * width * height + w * height + h] * scale[d];
				}
			}
		}
		for (u32 w = 0; w < width; w++) {
			for (u32 h = 0; h < height; h++) {			
				res[w] += sub_tensor[w * height + h] * vector[h];
			}
		}
	}
}

template <typename T>
void generate_data(T* tensor, T* vector, fp32* scale, u32 height, u32 width) {
	constexpr u32 dim = 4;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> fp_distr(0.0f, 1.0f);

	constexpr bool is_half_float = (sizeof(T) == sizeof(fp16));

	//file scale
	for (u32 d = 0; d < dim; d++) {
		scale[d] = fp_distr(gen);
	}
	alignas(alignof(__m128)) float ps[sizeof(__m128)/sizeof(fp32)];
	// fill vector
	for (u32 h = 0; h < height; h+=4) {
		ps[0] = fp_distr(gen);
		ps[1] = fp_distr(gen);
		ps[2] = fp_distr(gen);
		ps[3] = fp_distr(gen);
		if (is_half_float) {
			__m128i ph = _mm_cvtps_ph(*(__m128*)ps, 0);
			_mm_storeu_epi64((void*)&vector[h], ph);
		} else {
			_mm_storeu_ps((fp32*)&vector[h], *(__m128*)ps);
		}
	}
	// fill tensor
	for (u32 d = 0; d < dim; d++) {
		T* t_mat = (T*)&tensor[d * width * height];
		for (u32 w = 0; w < width; w++) {
			T* t_row = &t_mat[w * height];
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
}

template <typename T>
bool	fp_similar(T a, T b, T cmp = T(0.00001f)) { return abs(a - b) <= cmp; }

int main() {
	constexpr u32 h = 503; // 512
	constexpr u32 w = 288;
	constexpr u32 d = 4;

#ifdef USE_HALF_FLOAT
	typedef fp16 f_type;
#else	// USE_HALF_FLOAT
	typedef fp32 f_type;
#endif	// USE_HALF_FLOAT

#ifdef __AVX2__
	typedef __m256 m_reg;
#else // __AVX2__
	typedef __m128 m_reg;
#endif // __AVX2__

	f_type vector[h + sizeof(m_reg) / sizeof(f_type)];
	f_type*tensor = (f_type*)malloc(d * w * h * sizeof(f_type) + sizeof(m_reg));
	fp32 scale[d];

	generate_data(tensor, vector, scale, h, w);

	fp32 result_a[w];
	memset(result_a, 0, sizeof(result_a));
	vec_mat_mul(result_a, tensor, vector, scale, h, w);

	const u32 sub_tensor_sz_bytes = w * h * sizeof(fp32) + sizeof(m_reg);
	fp32* sub_tensor = (fp32*)malloc(sub_tensor_sz_bytes);
	memset(sub_tensor, 0, sub_tensor_sz_bytes);
	fp32 result_b[w];
	memset(result_b, 0, sizeof(result_b));
	simple_vec_mat_mul(result_b, tensor, sub_tensor, vector, scale, h, w);

	for (u32 i = 0; i < w; i++) {
		if (!fp_similar(result_a[i], result_b[i], 0.001f)) {
			printf("result_a[%d]: %.4f != result_b[%d]: %.4f\n", i, result_a[i], i, result_b[i]);
		}
	}

	free(tensor);
	free(sub_tensor);
	return 0;
}
