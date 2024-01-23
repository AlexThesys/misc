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

#define NUM_VALS 12
#define NON_ZERO_VALUES 12

#define EPS_4 0.0001f
#define EPS_5 0.00001f

template <class T>	
bool fp_zero(T	a, T cmp = T(EPS_5)) { return abs(a) <= cmp; }

template <u32 output_max>
fp32 map_into_range(fp32 value, fp32 input_max) {
	constexpr fp32 out_max = (fp32)output_max;
	if (fp_zero(input_max, EPS_5)) {
		return			0.0f;
	}
	else {
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

union packed_weights_32b {
	struct {
		u64 x0 : 12;
		u64 x1 : 10;
		u64 x2 : 10;
	};

	u32			u;
};

union packed_weights_64b {
	struct {
		// bits distribution - work in progress
		u64 x0 : 8;
		u64 x1 : 7;
		u64 x2 : 7;
		u64 x3 : 6;
		u64 x4 : 6;
		u64 x5 : 5;
		u64 x6 : 5;
		u64 x7 : 5;
		u64 x8 : 5;
		u64 x9 : 5;
		u64 x10 : 5;
	};

	u64			u;
};

enum max_weight_value {
	mwv_5_bit = 0x001F,
	mwv_6_bit = 0x003F,
	mwv_7_bit = 0x007F,
	mwv_8_bit = 0x00FF,
	mwv_9_bit = 0x01FF,
	mwv_10_bit = 0x03FF,
	mwv_11_bit = 0x07FF,
	mwv_12_bit = 0x0FFF,
};

inline void pack_weights_32b(const fp32 input[4], u32 size, u32& result) {
	size--;
	fp32 range = 1.0f;
	packed_weights_32b* res = (packed_weights_32b*)&result;

	res->x0 = (u16)roundf(map_into_range<mwv_12_bit>((size > 0) ? input[0] : 0.0f, range)); // maybe just an early exit at the top?
	range -= input[0];
	res->x1 = (u16)roundf(map_into_range<mwv_10_bit>((size > 1) ? input[1] : 0.0f, range));
	range -= input[1];
	res->x2 = (u16)roundf(map_into_range<mwv_10_bit>((size > 2) ? input[2] : 0.0f, range));
}

inline void unpack_weights_32b(fp32 result[4], const u32 input) {
	packed_weights_32b qvals = { 0 };
	qvals.u = input;
	fp32 range = 1.0f;

	result[0] = (fp32)qvals.x0;
	result[0] = map_from_range<mwv_12_bit>(result[0], range);
	range -= result[0];
	result[1] = (fp32)qvals.x1;
	result[1] = map_from_range<mwv_10_bit>(result[1], range);
	range -= result[1];
	result[2] = (fp32)qvals.x2;
	result[2] = map_from_range<mwv_10_bit>(result[2], range);
	range -= result[2];

	result[3] = range;
}

inline void pack_weights_64b(const fp32 input[12], u32 size, u64& result) {
	size--;
	fp32 range = 1.0f;
	packed_weights_64b* res = (packed_weights_64b*)&result;

	res->x0 = (u8)roundf(map_into_range<mwv_8_bit>((size > 0) ? input[0] : 0.0f, range)); // maybe just an early exit at the top?
	range -= input[0];
	res->x1 = (u8)roundf(map_into_range<mwv_7_bit>((size > 1) ? input[1] : 0.0f, range));
	range -= input[1];
	res->x2 = (u8)roundf(map_into_range<mwv_7_bit>((size > 2) ? input[2] : 0.0f, range));
	range -= input[2];
	res->x3 = (u8)roundf(map_into_range<mwv_6_bit>((size > 3) ? input[3] : 0.0f, range));
	range -= input[3];
	res->x4 = (u8)roundf(map_into_range<mwv_6_bit>((size > 4) ? input[4] : 0.0f, range));
	range -= input[4];
	res->x5 = (u8)roundf(map_into_range<mwv_5_bit>((size > 5) ? input[5] : 0.0f, range));
	range -= input[5];
	res->x6 = (u8)roundf(map_into_range<mwv_5_bit>((size > 6) ? input[6] : 0.0f, range));
	range -= input[6];
	res->x7 = (u8)roundf(map_into_range<mwv_5_bit>((size > 7) ? input[7] : 0.0f, range));
	range -= input[7];
	res->x8 = (u8)roundf(map_into_range<mwv_5_bit>((size > 8) ? input[8] : 0.0f, range));
	range -= input[8];
	res->x9 = (u8)roundf(map_into_range<mwv_5_bit>((size > 9) ? input[9] : 0.0f, range));
	range -= input[9];
	res->x10 = (u8)roundf(map_into_range<mwv_5_bit>((size > 10) ? input[10] : 0.0f, range));
}

inline void unpack_weights_64b(fp32 result[12], const u64 input) {
	packed_weights_64b qvals = { 0 };
	qvals.u = input;
	fp32 range = 1.0f;

	result[0] = (fp32)qvals.x0;
	result[0] = map_from_range<mwv_8_bit>(result[0], range);
	range -= result[0];
	result[1] = (fp32)qvals.x1;
	result[1] = map_from_range<mwv_7_bit>(result[1], range);
	range -= result[1];
	result[2] = (fp32)qvals.x2;
	result[2] = map_from_range<mwv_7_bit>(result[2], range);
	range -= result[2];
	result[3] = (fp32)qvals.x3;
	result[3] = map_from_range<mwv_6_bit>(result[3], range);
	range -= result[3];
	result[4] = (fp32)qvals.x4;
	result[4] = map_from_range<mwv_6_bit>(result[4], range);
	range -= result[4];
	result[5] = (fp32)qvals.x5;
	result[5] = map_from_range<mwv_5_bit>(result[5], range);
	range -= result[5];
	result[6] = (fp32)qvals.x6;
	result[6] = map_from_range<mwv_5_bit>(result[6], range);
	range -= result[6];
	result[7] = (fp32)qvals.x7;
	result[7] = map_from_range<mwv_5_bit>(result[7], range);
	range -= result[7];
	result[8] = (fp32)qvals.x8;
	result[8] = map_from_range<mwv_5_bit>(result[8], range);
	range -= result[8];
	result[9] = (fp32)qvals.x9;
	result[9] = map_from_range<mwv_5_bit>(result[9], range);
	range -= result[9];
	result[10] = (fp32)qvals.x10;
	result[10] = map_from_range<mwv_5_bit>(result[10], range);
	range -= result[10];

	result[11] = range;
}

int main() {
	static_assert(NUM_VALS >= NON_ZERO_VALUES, "Incorrect input aruments");

	alignas(alignof(__m128))
	fp32 values[NUM_VALS];
	fp32 values_b[NUM_VALS];
	//fp32 values_c[NUM_VALS];
	fp64 total_accum_orig = 0.0f;
	fp64 total_accum_fixed = 0.0f;
	fp64 total_accum_var = 0.0f;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> fp_distr(0.0f, 1.0f);

	for (u32 j = 0; j < NUM_ITER; j++) {
		fp64 accum_orig = 0.0f;
		fp64 accum_fixed = 0.0f;
		fp64 accum_var = 0.0f;
		fp64 total_weight = 0.0f;

		printf("****Iteration #%d****\n\n", j);
		// generate random values
		for (int i = 0; i < NON_ZERO_VALUES; i++) {
			values[i] = fp_distr(gen);
			total_weight += values[i];
		}
		for (int i = NON_ZERO_VALUES; i < NUM_VALS; i++) {
			values[i] = 0;
			total_weight += values[i];
		}

		std::sort(values, values + NUM_VALS, [](fp32 a, fp32 b) {return a > b; });
		// normalize
		total_weight = 1.0f / total_weight;
		puts("==== Original Values ====");
		for (int i = 0; i < NUM_VALS; i++) {
			values[i] *= total_weight;
			accum_orig += (fp64)values[i];
			printf("%.6f ", values[i]);
		}

		puts("\n==== Fixed Range Quantization ====");
		// original quantization
		for (int i = 0; i < NUM_VALS; i++) {
			fp32 f = values[i];
			const u8 q = (u8)roundf(f * fp32(mwv_8_bit));
			f = (fp32)q / mwv_8_bit;
			accum_fixed += (fp64)f;
			printf("%.6f ", f);
		}

		memcpy(values_b, values, sizeof(values_b));
		//memcpy(values_c, values, sizeof(values_c));

		puts("\n==== Variable Range Quantization ====");
		// variable range quantization A
		u64 qval;
		pack_weights_64b(values, NUM_VALS, qval);
		unpack_weights_64b(values, qval);
		for (u32 i = 0; i < NUM_VALS; i++) {
			accum_var += (fp64)values[i];
			printf("%.6f ", values[i]);
		}

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
