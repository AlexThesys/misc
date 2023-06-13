#include <stdint.h>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <random>

typedef float fp32;
typedef uint32_t u32;
typedef int32_t s32;
typedef uint8_t fp8;
typedef uint8_t u8;
typedef double fp64;
typedef uint16_t u16;


/*
	only numbers in the range [0, 1] can be represented
*/

// e = 4b, m = 4b

u32 float_to_quater_a(fp32 f) {
	assert(f <= 1.0f && f >= 0.0f);
	s32 e = (*(u32*)&f & 0x7f800000) >> 0x17;
	e = e - 0x7F;
	assert(e <= 0);
	e = ~e + 1; // negate
	e &= 0x0F;
	const u32 q = (e << 0x04) | ((*(u32*)&f & 0x007FFFFF) >> 0x13);
	return *(u32*)&q;
}

fp32 quater_to_float_a(u32 q) {
	const u32 e = 0x7F - ((q & 0xF0) >> 0x04);
	const u32 f = (e << 0x17) | ((q & 0x0F) << 0x13);
	return *(fp32*)&f;
}

// e = 3b, m = 5b

u32 float_to_quater_a2(fp32 f) {
	assert(f <= 1.0f && f >= 0.0f);
	s32 e = (*(u32*)&f & 0x7f800000) >> 0x17;
	e = e - 0x7F;
	assert(e <= 0);
	e = ~e + 1; // negate
	e &= 0x07;
	const u32 q = (e << 0x05) | ((*(u32*)&f & 0x007FFFFF) >> 0x12);
	return *(u32*)&q;
}

fp32 quater_to_float_a2(u32 q) {
	const u32 e = 0x7F - ((q & 0xE0) >> 0x05);
	const u32 f = (e << 0x17) | ((q & 0x1F) << 0x12);
	return *(fp32*)&f;
}

typedef struct { u16 s : 9; } fp9;

// e = 4b, m = 5b

u32 float_to_quater_b(fp32 f) {
	assert(f <= 1.0f && f >= 0.0f);
	s32 e = (*(u32*)&f & 0x7f800000) >> 0x17;
	e = e - 0x7f;
	assert(e <= 0);
	e = ~e + 1; // negate
	e &= 0x0F;
	const u32 q = (e << 0x05) | ((*(u32*)&f & 0x007FFFFF) >> 0x12);
	return *(u32*)&q;
}

fp32 quater_to_float_b(u32 q) {
	const u32 e = 0x7F - ((q & 0x01E0) >> 0x05);
	const u32 f = (e << 0x17) | ((q & 0x1F) << 0x12);
	return *(fp32*)&f;
}

typedef struct { u16 s : 10; } fp10;

// e = 4b, m = 6b

u32 float_to_quater_c(fp32 f) {
	assert(f <= 1.0f && f >= 0.0f);
	s32 e = (*(u32*)&f & 0x7f800000) >> 0x17;
	e = e - 0x7f;
	assert(e <= 0);
	e = ~e + 1; // negate
	e &= 0x0f;
	const u32 q = (e << 0x06) | ((*(u32*)&f & 0x007FFFFF) >> 0x11);
	return *(u32*)&q;
}

fp32 quater_to_float_c(u32 q) {
	const u32 e = 0x7F - ((q & 0x03C0) >> 0x06);
	const u32 f = (e << 0x17) | ((q & 0x3F) << 0x11);
	return *(fp32*)&f;
}

typedef struct { u16 s : 12; } fp12;

// e = 4b, m = 8b

u32 float_to_quater_d(fp32 f) {
	assert(f <= 1.0f && f >= 0.0f);
	s32 e = (*(u32*)&f & 0x7f800000) >> 0x17;
	e = e - 0x7f;
	assert(e <= 0);
	e = ~e + 1; // negate
	e &= 0x0F;
	const u32 q = (e << 0x08) | ((*(u32*)&f & 0x007FFFFF) >> 0x0F);
	return *(u32*)&q;
}

fp32 quater_to_float_d(u32 q) {
	const u32 e = 0x7F - ((q & 0x0F00) >> 0x08);
	const u32 f = (e << 0x17) | ((q & 0xFF) << 0x0F);
	return *(fp32*)&f;
}

typedef u32 fp16;

// e = 5b, m = 11b

u32 float_to_quater_e(fp32 f) {
	assert(f <= 1.0f && f >= 0.0f);
	s32 e = (*(u32*)&f & 0x7f800000) >> 0x17;
	e = e - 0x7f;
	assert(e <= 0);
	e = ~e + 1; // negate
	e &= 0x1F;
	const u32 q = (e << 0x0B) | ((*(u32*)&f & 0x007FFFFF) >> 0x0C);
	return *(u32*)&q;
}

fp32 quater_to_float_e(u32 q) {
	const u32 e = 0x7F - ((q & 0xF800) >> 0x0B);
	const u32 f = (e << 0x17) | ((q & 0x07FF) << 0x0C);
	return *(fp32*)&f;
}

fp8 float_to_byte(fp32 f) {
	return (fp8)roundf(f *  255.0f);
}

fp32 byte_to_float(fp8 q) {
	return (fp32)q / 255.0f;
}

#define EPS_5 0.00001f

u32 quantize(fp32 val, fp32 min, fp32 max, u32 bit_count, u32 adj = 1)
{
	fp32 q = (val - min) / (max - min);
	return			floor(q * fp32((1 << bit_count) - adj) + .5f);
}
fp32 dequantize(u32 val, fp32 min, fp32 max, u32 bit_count, u32 adj = 1)
{
	fp32 A = (fp32)((fp64(val) * (max - min)) / fp64((1 << bit_count) - adj) + min);
	assert((A >= min - EPS_5) && (A <= max + EPS_5));
	return			A;
}

#define NUM_ITER 50
#define SKEW 10.0f	// 10.0f

int main() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> fp_distr(0.0f, 1.0f);
	puts("================");
	fp32 accum_orig = 0.0f;
	fp32 accum_a = 0.0f;
	fp32 accum_a2 = 0.0f;
	fp32 accum_b = 0.0f;
	fp32 accum_c = 0.0f;
	fp32 accum_d = 0.0f;
	fp32 accum_h = 0.0f;
	fp32 accum_e = 0.0f;
	fp32 accum_f = 0.0f;
	fp32 accum_g = 0.0f;
	for (int i = 0; i < NUM_ITER; i++) {
		const fp32 f = fp_distr(gen) / SKEW;
		printf("Original value\t\t %0.6f\n", f);
		accum_orig += f;

		fp8 q = float_to_quater_a(f);
		fp32 x = quater_to_float_a(q);
		printf("A: 8b fp value\t\t %0.6f\n", x);
		accum_a += x;

		q = float_to_quater_a2(f);
		x = quater_to_float_a2(q);
		printf("A2: 8b fp value\t\t %0.6f\n", x);
		accum_a2 += x;

		fp9 w;
		w.s = float_to_quater_b(f);
		x = quater_to_float_b(w.s);
		printf("B: 9b fp value\t\t %0.6f\n", x);
		accum_b += x;

		fp10 z; 
		z.s = float_to_quater_c(f);
		x = quater_to_float_c(z.s);
		printf("C: 10b fp value\t\t %0.6f\n", x);
		accum_c += x;

		fp12 v;
		v.s = float_to_quater_d(f);
		x = quater_to_float_d(v.s);
		printf("D: 12b fp value\t\t %0.6f\n", x);
		accum_d += x;

		fp16 h;
		h = float_to_quater_e(f);
		x = quater_to_float_e(h);
		printf("E: 16b fp value\t\t %0.6f\n", x);
		accum_e += x;

		w.s = quantize(f, 0.0f, 1.0f, 9);
		x = dequantize(w.s, 0.0f, 1.0f, 9);
		printf("F: 9b quantized value\t %0.6f\n", x);
		accum_f += x;

		z.s = quantize(f, 0.0f, 1.0f, 10);
		x = dequantize(z.s, 0.0f, 1.0f, 10);
		printf("G: 10b quantized value\t %0.6f\n", x);
		accum_g += x;

		q = float_to_byte(f);
		x = byte_to_float(q);
		printf("H: 8b quantized value\t %0.6f\n", x);
		accum_h += x;

		puts("================");
	}
	printf("Accumulated error A:\t %.6f\n", abs(accum_orig - accum_a));
	printf("Accumulated error A2:\t %.6f\n", abs(accum_orig - accum_a2));
	printf("Accumulated error B:\t %.6f\n", abs(accum_orig - accum_b));
	printf("Accumulated error C:\t %.6f\n", abs(accum_orig - accum_c));
	printf("Accumulated error D:\t %.6f\n", abs(accum_orig - accum_d));
	printf("Accumulated error E:\t %.6f\n", abs(accum_orig - accum_e));
	printf("Accumulated error F:\t %.6f\n", abs(accum_orig - accum_f));
	printf("Accumulated error G:\t %.6f\n", abs(accum_orig - accum_g));
	printf("Accumulated error H:\t %.6f\n", abs(accum_orig - accum_h));
}