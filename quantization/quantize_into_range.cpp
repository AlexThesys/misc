#include <stdint.h>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <random>
#include <algorithm>
#include <functional>
#include <smmintrin.h>

typedef float fp32;
typedef uint32_t u32;
typedef int32_t s32;
typedef uint8_t fp8;
typedef uint8_t u8;
typedef double fp64;
typedef uint16_t u16;
typedef uint64_t u64;


#define NUM_ITER 10 

#define NUM_VALS 8 

#define EPS_4 0.0001f
#define EPS_5 0.00001f

#define MAX_8_BIT 0X00FF
#define MAX_9_BIT 0X01FF
#define MAX_10_BIT 0X03FF



union packed_64b {
	struct {
		u64 x0 : 10;
		u64 x1 : 9;
		u64 x2 : 9;
		u64 x3 : 9;
		u64 x4 : 9;
		u64 x5 : 9;
		u64 x6 : 9;
	};

	u64 u;
};

template <class T>	
bool fp_zero(T	a, T cmp = T(EPS_5)) { return abs(a) <= cmp; }

template <u32 output_max>
fp32 map_into_range(fp32 value, fp32 input_max) {
	constexpr fp32 out_max = (fp32)output_max;
	if (fp_zero(input_max, EPS_5)) {
		return			0.0f;
	} else {
		fp32 out_val = value / input_max * out_max;
		return			out_val;
	}
}

template <u32 input_max>
fp32 map_from_range(fp32 value, fp32 output_max) {
	constexpr fp32 input_max_rec = 1.0f / (fp32)input_max;
		fp32 out_val = value * input_max_rec * output_max;
		return			out_val;
}

fp32 map_into_range(fp32 value, fp32 input_max, fp32 output_max) {
	if (fp_zero(input_max, EPS_5)) {
		return			0.0f;
	}
	else {
		fp32 out_val = value / input_max * output_max;
		return			out_val;
	}
}

fp32 map_from_range(fp32 value, fp32 input_max, fp32 output_max) {
	const fp32 input_max_rec = 1.0f / (fp32)input_max;
	fp32 out_val = value * input_max_rec * output_max;
	return			out_val;
}

u64 quantize_variable_range_a(const fp32* f, u32 num_vals) {
	fp32 range = 1.0f;
	u64 res;
	u8* u = (u8*)&res;
	for (u32 i = 0; i < num_vals; i++) {
		u[i] = (u8)roundf(map_into_range<MAX_8_BIT>(f[i], range));
		range -= f[i];
	}
	return res;
}

fp32 dequantize_variable_range_a(fp32* f, u64 u, u32 num_vals) {
	fp32 accum = 0.0f;
	fp32 range = 1.0f;
	const u8* qval = (const u8*)&u;
	for (u32 i = 0; i < num_vals; i++) {
		const fp32 f_old = (fp32)qval[i];
		f[i] = map_from_range<MAX_8_BIT>(f_old, range);
		range -= f[i];
		printf("%.6f ", f[i]);
		accum += f[i];
	}
	return accum;
}

u64 quantize_variable_range_b(const fp32* f, u32 num_vals) {
	assert(num_vals <= 8);
	num_vals--;
	fp32 range = 1.0f;
	packed_64b res;

	res.x0 = (u16)roundf(map_into_range<MAX_10_BIT>(f[0], range));
	range -= f[0];
	res.x1 = (u16)roundf(map_into_range<MAX_9_BIT>(f[1], range));
	range -= f[1];
	res.x2 = (u16)roundf(map_into_range<MAX_9_BIT>(f[2], range));
	range -= f[2];
	res.x3 = (u16)roundf(map_into_range<MAX_9_BIT>(f[3], range));
	range -= f[3];
	res.x4 = (u16)roundf(map_into_range<MAX_9_BIT>(f[4], range));
	range -= f[4];
	res.x5 = (u16)roundf(map_into_range<MAX_9_BIT>(f[5], range));
	range -= f[5];
	res.x6 = (u16)roundf(map_into_range<MAX_9_BIT>(f[6], range));

	return res.u;
}

void dequantize_variable_range_b(fp32* f, u64 u, u32 num_vals) {
	assert(num_vals <= 8);
	num_vals--;
	fp32 range = 1.0f;
	packed_64b qvals = {0};
	qvals.u = u;
	fp32 f_old[NUM_VALS-1];
	f_old[0] = (fp32)qvals.x0;
	f_old[1] = (fp32)qvals.x1;
	f_old[2] = (fp32)qvals.x2;
	f_old[3] = (fp32)qvals.x3;
	f_old[4] = (fp32)qvals.x4;
	f_old[5] = (fp32)qvals.x5;
	f_old[6] = (fp32)qvals.x6;

	f[0] = map_from_range<MAX_10_BIT>(f_old[0], range);
	range -= f[0];
	printf("%.6f ", f[0]);
	for (u32 i = 1, sz = num_vals; i < sz; i++) {
		f[i] = map_from_range<MAX_9_BIT>(f_old[i], range);
		range -= f[i];
		printf("%.6f ", f[i]);
	}

	f[num_vals] = range;
	printf("%.6f ", f[num_vals]);
}

int main() {
	alignas(alignof(__m128))
	fp32 values[NUM_VALS];
	fp32 values_b[NUM_VALS];
	//fp32 values_c[NUM_VALS];
	fp32 total_accum_orig = 0.0f;
	fp32 total_accum_fixed = 0.0f;
	fp32 total_accum_var = 0.0f;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> fp_distr(0.0f, 1.0f);

	for (u32 j = 0; j < NUM_ITER; j++) {
		fp32 accum_orig = 0.0f;
		fp32 accum_fixed = 0.0f;
		fp32 accum_var = 0.0f;
		fp32 total_weight = 0.0f;

		printf("****Iteration #%d****\n\n", j);
		// generate random values
		for (int i = 0; i < NUM_VALS; i++) {
			values[i] = fp_distr(gen);
			total_weight += values[i];
		}

		std::sort(values, values + NUM_VALS, [](fp32 a, fp32 b) {return a > b; });
		// normalize
		total_weight = 1.0f / total_weight;
		puts("==== Original Values ====");
		for (int i = 0; i < NUM_VALS; i++) {
			values[i] *= total_weight;
			accum_orig += values[i];
			printf("%.6f ", values[i]);
		}

		puts("\n==== Fixed Range Quantization ====");
		// original quantization
		for (int i = 0; i < NUM_VALS; i++) {
			fp32 f = values[i];
			const u8 q = (u8)roundf(f * fp32(MAX_8_BIT));
			f = (fp32)q / MAX_8_BIT;
			accum_fixed += f;
			printf("%.6f ", f);
		}

		memcpy(values_b, values, sizeof(values_b));
		memcpy(values_c, values, sizeof(values_c));

		puts("\n==== Variable Range Quantization A ====");
		// variable range quantization A
		u64 qval = quantize_variable_range_a(values, NUM_VALS);
		accum_var = dequantize_variable_range_a(values, qval, NUM_VALS);

		puts("\n==== Variable Range Quantization B ====");
		// variable range quantization B
		qval = 0;
		qval = quantize_variable_range_b(values_b, NUM_VALS);
		dequantize_variable_range_b(values_b, qval, NUM_VALS);

		printf("\n\nAccumulated error fixed range:\t\t %.7f\n", abs(accum_orig - accum_fixed));
		printf("Accumulated error variable range:\t %.7f\n\n", abs(accum_orig - accum_var));

		total_accum_orig += accum_orig;
		total_accum_fixed += accum_fixed;
		total_accum_var += accum_var;
	}
	puts("\n*****************");
	printf("Total accumulated error fixed range:\t %.7f\n", abs(total_accum_orig - total_accum_fixed));
	printf("Total accumulated error variable range:\t %.7f\n\n", abs(total_accum_orig - total_accum_var));
}
