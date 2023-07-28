#include "benchmark/benchmark.h"

#include <stdint.h>
#include <assert.h>
#include <memory>
#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <random>
#include <string.h>

#include <windows.h>

typedef float fp32;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint16_t fp16;
typedef int32_t s32;
typedef uint32_t u32;
typedef uint64_t u64;

#define USE_HALF_FLOAT

constexpr u32 h = 0x400;
constexpr u32 w = 0x400;

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

template <typename T>
void vec_mat_mul(fp32* res, const T* tensor, const fp32* vector, u32 height, u32 width) {	// height == row, width = col
	constexpr bool is_half_float = (sizeof(T) == sizeof(fp16));
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
			__m128 accum = _mm_add_ps(hi, *(__m128*) & accum_256);
			*out = horizontal_add(accum);
		}
	} else {
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
			__m128 accum = _mm_add_ps(hi, *(__m128*) & accum_256);
			*out = horizontal_add(accum);
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
				_mm_storeu_si64((void*)&t_row[h], ph);
			}
			else {
				_mm_storeu_ps((fp32*)&t_row[h], *(__m128*)ps);
			}
		}
	}
}

static void vec_mat_mul(benchmark::State& state) {
	const u32 height = h;
	const u32 width = w;

#ifdef USE_HALF_FLOAT
	typedef fp16 f_type;
#else	// USE_HALF_FLOAT
	typedef fp32 f_type;
#endif	// USE_HALF_FLOAT

	typedef __m256 m_reg;

	alignas(alignof(m_reg)) fp32 vector[h];
	f_type* tensor = (f_type*)malloc(w * h * sizeof(f_type) + sizeof(m_reg));
	generate_data(tensor, vector, h, w);
	alignas(alignof(m_reg)) fp32 result_a[w];
    for (auto _ : state) {     
		vec_mat_mul(result_a, tensor, vector, height, width);
        benchmark::DoNotOptimize(result_a);
    }
	free(tensor);
}

////////////////////////////////

#define THREADCOUNT 8
volatile int g_stop_threads;
HANDLE ghSemaphore;
HANDLE ghEvent[THREADCOUNT];
volatile long g_counter;

#ifdef USE_HALF_FLOAT
typedef fp16 f_type;
#else	// USE_HALF_FLOAT
typedef fp32 f_type;
#endif	// USE_HALF_FLOAT

struct in_out_data {
	fp32* result;
	const f_type* tensor;
	const fp32* vector;
	const volatile int* num_cols;
	volatile int block_size;
	int tensor_offset;
	int t_id;
};

#define _max(x, y) (x) > (y) ? (x) : (y)
#define _min(x, y) (x) < (y) ? (x) : (y)

DWORD WINAPI MyThreadFunction(LPVOID lpParam) {
	const in_out_data* io_data = (in_out_data*)lpParam;
	fp32* result = io_data->result;
	const f_type* tensor = io_data->tensor;
	const fp32* vector = io_data->vector;
	const int t_id = io_data->t_id;
	while (true) {
		WaitForSingleObject(ghEvent[t_id], INFINITE);
		if (g_stop_threads)
			break;
		const int h = *io_data->num_cols;
		const int offset = io_data->tensor_offset;
		const f_type* tensor_begin = tensor + h * offset;
		const int block_size = io_data->block_size;
		fp32* res = result + offset;
		if (block_size)
			vec_mat_mul(res, tensor_begin, vector, h, block_size);

		InterlockedIncrement(&g_counter);
		ReleaseSemaphore(ghSemaphore, 1, NULL);
	}
	return 0;
}

static void vec_mat_mul_mt(benchmark::State& state) {
	g_stop_threads = 0;

	typedef __m256 m_reg;
	volatile int height = h;
	volatile int width = w;

	alignas(alignof(m_reg)) fp32 vector[h];
	f_type* tensor = (f_type*)malloc(w * h * sizeof(f_type) + sizeof(m_reg));
	generate_data(tensor, vector, h, w);
	alignas(alignof(m_reg)) fp32 result_a[w];

	in_out_data io_data[THREADCOUNT];
	////////////////////////
	HANDLE aThread[THREADCOUNT];
	DWORD dwThreadID;
	ghSemaphore = CreateSemaphore(NULL, 0, THREADCOUNT, NULL);
	for (int i = 0; i < THREADCOUNT; i++) {
		ghEvent[i] = CreateEvent(NULL, FALSE, FALSE, NULL);

		io_data[i].result = result_a;
		io_data[i].tensor = tensor;
		io_data[i].vector = vector;
		io_data[i].num_cols = &height;
		io_data[i].block_size = 0;
		io_data[i].t_id = i;
		aThread[i] = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)MyThreadFunction, (io_data + i), 0, &dwThreadID);
	}

	//////////////////////////////

	for (auto _ : state) {
		// but there could've been different each time
		height = h;
		width = w;
		const int max_threads = _min(THREADCOUNT, width);
		const int block_size = width / max_threads;
		int tail = width - block_size * max_threads;
		int offset_accum = 0;
		for (int i = 0; i < max_threads; i++) {
			io_data[i].block_size = block_size + ((tail-- > 0) ? 1 : 0);
			io_data[i].tensor_offset = offset_accum;
			offset_accum += io_data[i].block_size;
		}
		for (int i = max_threads; i < THREADCOUNT; i++)
			io_data[i].block_size = 0;

		for (int i = 0; i < THREADCOUNT; i++)
			SetEvent(ghEvent[i]);

		while (g_counter < THREADCOUNT)
			WaitForSingleObject(ghSemaphore, INFINITE);

		g_counter = 0;
		benchmark::DoNotOptimize(result_a);
	}

	////////////////////////////
	g_stop_threads = 1;
	_mm_sfence();
	for (int i = 0; i < THREADCOUNT; i++)
		SetEvent(ghEvent[i]);

	WaitForMultipleObjects(THREADCOUNT, aThread, TRUE, INFINITE);
	for (int i = 0; i < THREADCOUNT; i++)
		CloseHandle(aThread[i]);

	CloseHandle(ghSemaphore);
	for (int i = 0; i < THREADCOUNT; i++)
		CloseHandle(ghEvent[i]);
	////////////////////////////

	free(tensor);
}

BENCHMARK(vec_mat_mul);
BENCHMARK(vec_mat_mul_mt);

BENCHMARK_MAIN();