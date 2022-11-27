#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <random>

#include <xmmintrin.h>
#include <emmintrin.h>

#include "benchmark/benchmark.h"


__m128 log_a(__m128 val);
__m128 pow_a(__m128 val);
__m128 log_b(__m128 val);
__m128 log10v(__m128 val);
__m128 pow_b(__m128 val);

typedef uint32_t u32;

enum powv_n {
    powv_10 = 0,
    powv_2 = 1,
};

template <powv_n>
__m128 powv(__m128 val);

// validate correctness
//int main()
//{
//    const __m128 init = _mm_set_ps(0.001f, 0.002f, 0.003f, 0.004f);   
//    alignas(16) float result_a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
//    alignas(16) float result_b[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
//    std::default_random_engine generator;
//    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
//    for (int i = 0; i < 2; i++) {
//
//        const __m128 val = _mm_set_ps   (distribution(generator), distribution(generator),
//                                        distribution(generator), distribution(generator));
//        __m128 res = pow_a(val);
//        _mm_store_ps(result_a, res);
//        res = pow_b(val);
//        _mm_store_ps(result_b, res);
//        for (int j = 0; j < 4; j++) {
//            if (result_a[j] != result_b[j])
//                puts("fail!");
//        }
//        res = log_a(val);
//        _mm_store_ps(result_a, res);
//        res = log_b(val);
//        _mm_store_ps(result_b, res);
//        for (int j = 0; j < 4; j++) {
//            if (result_a[j] != result_b[j])
//                puts("fail!");
//        }
//    }
//
//    return 0;
//}

__m128 log_a(__m128 val) {
    alignas(16) float temp[4];
    _mm_store_ps(temp, val);
    temp[0] = log10f(temp[0]);
    temp[1] = log10f(temp[1]);
    temp[2] = log10f(temp[2]);
    temp[3] = log10f(temp[3]);
    __m128 res = _mm_load_ps(temp);
    return res;
}

__m128 pow_a(__m128 val) {
    alignas(16) float temp[4];
    _mm_store_ps(temp, val);
    temp[0] = powf(10, temp[0]);
    temp[1] = powf(10, temp[1]);
    temp[2] = powf(10, temp[2]);
    temp[3] = powf(10, temp[3]);
    val = _mm_load_ps(temp);
    return val;
}

__m128 log_b(__m128 val) {
    return log10v(val);
}

__m128 pow_b(__m128 val) {
    return powv<powv_10>(val);
}

union m128 {
    __m128 f;
    __m128d d;
    __m128i i;
};

__m128 log10v(__m128 val) {
    const __m128 eps = _mm_set1_ps(0.000000001f);
    const __m128i mantissa_mask = _mm_set1_epi32(0x007fffff);
    const __m128i sqrt_2_addition = _mm_set1_epi32(0x004afb10);
    const __m128i f_one = _mm_set1_epi32(0x3f800000);
    const __m128i sqrt_2_mask = _mm_set1_epi32(0x00800000);
    const __m128i exp_bias = _mm_set1_epi32(0x0000007f);
    const __m128i f_inv_ln10 = _mm_set1_epi32(0x3ede5bd9);
    const __m128i f_b0 = _mm_set1_epi32(0x3e943d93);
    const __m128i f_b1 = _mm_set1_epi32(0x3e319274);
    const __m128i f_b2 = _mm_set1_epi32(0x3e096bb1);
    const __m128i f_lg2 = _mm_set1_epi32(0x3e9a209b);

    m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;
    xmm0.f = _mm_max_ps(val, eps);  // input
    xmm1.i = _mm_and_si128(xmm0.i, mantissa_mask);
    xmm2.i = _mm_add_epi32(sqrt_2_addition, xmm1.i);
    xmm2.i = _mm_and_si128(xmm2.i, sqrt_2_mask);
    xmm3.i = _mm_xor_si128(f_one, xmm2.i);
    xmm2.i = _mm_srai_epi32(xmm2.i, 0x17);
    xmm1.i = _mm_or_si128(xmm1.i, xmm3.i);
    xmm0.i = _mm_srai_epi32(xmm0.i, 0x17);
    xmm4.f = _mm_add_ps(xmm1.f, *(__m128*)&f_one);
    xmm1.f = _mm_sub_ps(xmm1.f, *(__m128*)&f_one);
    xmm0.i = _mm_sub_epi32(xmm0.i, exp_bias);
    xmm0.i = _mm_add_epi32(xmm0.i, xmm2.i);
    xmm0.f = _mm_cvtepi32_ps(xmm0.i);
    xmm2.f = _mm_mul_ps(*(__m128*)&f_inv_ln10, xmm1.f);
    xmm1.f = _mm_div_ps(xmm1.f, xmm4.f);
    xmm4.f = xmm1.f;
    xmm1.f = _mm_mul_ps(xmm1.f, xmm1.f);
    xmm3.f = xmm1.f;
    xmm1.f = _mm_mul_ps(xmm1.f, xmm1.f);
    xmm5.f = _mm_mul_ps(xmm1.f, *(__m128*)&f_b2);
    xmm1.f = _mm_mul_ps(xmm1.f, *(__m128*)&f_b1);
    xmm5.f = _mm_add_ps(xmm5.f, *(__m128*)&f_b0);
    xmm5.f = _mm_mul_ps(xmm5.f, xmm3.f);
    xmm0.f = _mm_mul_ps(xmm0.f, *(__m128*) & f_lg2);
    xmm5.f = _mm_add_ps(xmm5.f, xmm1.f);
    xmm5.f = _mm_sub_ps(xmm5.f, xmm2.f);
    xmm5.f = _mm_mul_ps(xmm5.f, xmm4.f);
    xmm5.f = _mm_add_ps(xmm5.f, xmm2.f);
    xmm0.f = _mm_add_ps(xmm0.f, xmm5.f);
    return xmm0.f;
}

struct alignas(alignof(__m128)) powv_lookup_table {
    u32 d_k_3_5[4];
    u32 d_k_110[4];
    u32 d_k_130[4];
    u32 d_k_140[4];
    u32 d_k_150[4];
    u32 d_k_160[4];
    u32 d_k_170[4];
    u32 d_k_180[4];
    u32 d_k_190[4];
    u32 d_k_1A0[4];
};

static const powv_lookup_table powv_d_k[2] = 
{
    {
        {0x0979A2A4, 0x400A934F, 0x0979A2A4, 0x400A934F},
        {0x00000000, 0x43380000, 0x00000000, 0x43380000},
        {0x000003f2, 0x00000000, 0x000003f2, 0x00000000},
        {0x213bde9f, 0x401b6eb4, 0x213bde9f, 0x401b6eb4},
        {0x348d1e8d, 0x4010bf17, 0x348d1e8d, 0x4010bf17},
        {0x50304bbe, 0x402a6a2a, 0x50304bbe, 0x402a6a2a},
        {0xa22d3a94, 0x4030a032, 0xa22d3a94, 0x4030a032},
        {0xbb3fa39c, 0xc002807e, 0xbb3fa39c, 0xc002807e},
        {0x293ab434, 0x403dad2e, 0x293ab434, 0x403dad2e},
        {0x371366f6, 0x00041d33, 0x371366f6, 0x00041d33},
    },
    {
        {0x00000000, 0x3FF00000, 0x00000000, 0x3FF00000},
        {0x00000000, 0x43380000, 0x00000000, 0x43380000},
        {0x000003f2, 0x00000000, 0x000003f2, 0x00000000},
        {0x213bde9f, 0x401b6eb4, 0x213bde9f, 0x401b6eb4},
        {0x348d1e8d, 0x4010bf17, 0x348d1e8d, 0x4010bf17},
        {0x50304bbe, 0x402a6a2a, 0x50304bbe, 0x402a6a2a},
        {0xa22d3a94, 0x4030a032, 0xa22d3a94, 0x4030a032},
        {0xbb3fa39c, 0xc002807e, 0xbb3fa39c, 0xc002807e},
        {0x293ab434, 0x403dad2e, 0x293ab434, 0x403dad2e},
        {0x371366f6, 0x00041d33, 0x371366f6, 0x00041d33},
    },
};

template <powv_n N>
__m128 powv(__m128 val) {
    static_assert(N == powv_2 || N == powv_10, "Only powers of 2 and 10 are currently supported.");
    const powv_lookup_table *d_k = &powv_d_k[N];
    const __m128i *d_k_3_5   = (__m128i*)&d_k->d_k_3_5; 
    const __m128i *d_k_110   = (__m128i*)&d_k->d_k_110;
    const __m128i *d_k_130   = (__m128i*)&d_k->d_k_130;
    const __m128i *d_k_140   = (__m128i*)&d_k->d_k_140;
    const __m128i *d_k_150   = (__m128i*)&d_k->d_k_150;
    const __m128i *d_k_160   = (__m128i*)&d_k->d_k_160;
    const __m128i *d_k_170   = (__m128i*)&d_k->d_k_170;
    const __m128i *d_k_180   = (__m128i*)&d_k->d_k_180;
    const __m128i *d_k_190   = (__m128i*)&d_k->d_k_190;
    const __m128i *d_k_1A0   = (__m128i*)&d_k->d_k_1A0;

    m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    xmm2.d  = _mm_cvtps_pd(val);
    xmm6.i  = _mm_srli_si128(*(__m128i*)&val, 0x08); // logical shift right by 8 bytes
    xmm6.d  = _mm_cvtps_pd(xmm6.f);
    xmm3.d  = _mm_mul_pd(*(__m128d*)d_k_3_5, xmm2.d);
    xmm5.d  = _mm_mul_pd(*(__m128d*)d_k_3_5, xmm6.d);
    xmm0.d  = _mm_add_pd(*(__m128d*)d_k_110, xmm3.d);
    xmm4.d  = _mm_sub_pd(*(__m128d*)d_k_110, xmm0.d);
    xmm3.d  = _mm_add_pd(xmm3.d, xmm4.d);
    xmm0.i  = _mm_add_epi32(xmm0.i, *d_k_130);
    xmm1.d  = _mm_add_pd(*(__m128d*)d_k_110, xmm5.d);
    xmm6.d  = _mm_sub_pd(*(__m128d*)d_k_110, xmm1.d);
    xmm5.d  = _mm_add_pd(xmm5.d, xmm6.d);
    xmm7.d  = _mm_add_pd(*(__m128d*)d_k_140, xmm3.d);
    xmm1.i  = _mm_add_epi32(xmm1.i, *d_k_130);
    xmm6.d  = xmm5.d;
    xmm7.d  = _mm_mul_pd(xmm7.d, xmm3.d);
    xmm5.d  = _mm_add_pd(xmm5.d, *(__m128d*)d_k_150);
    xmm4.d  = _mm_add_pd(xmm3.d, *(__m128d*)d_k_150);
    xmm7.d  = _mm_add_pd(xmm7.d, *(__m128d*)d_k_160);
    xmm5.d  = _mm_mul_pd(xmm5.d, xmm6.d);
    xmm4.d  = _mm_mul_pd(xmm4.d, xmm3.d);
    xmm2.d  = _mm_add_pd(xmm6.d, *(__m128d*)d_k_140);
    xmm0.i  = _mm_slli_epi64(xmm0.i, 0x34);
    xmm5.d  = _mm_add_pd(xmm5.d, *(__m128d*)d_k_170);
    xmm4.d  = _mm_add_pd(xmm4.d, *(__m128d*)d_k_170);
    xmm2.d  = _mm_mul_pd(xmm2.d, xmm6.d);
    xmm7.d  = _mm_mul_pd(xmm7.d, xmm4.d);
    xmm2.d  = _mm_add_pd(xmm2.d, *(__m128d*)d_k_160);
    xmm4.d  = xmm3.d;
    xmm5.d  = _mm_mul_pd(xmm5.d, xmm2.d);
    xmm3.d  = _mm_add_pd(xmm3.d, *(__m128d*)d_k_180);
    xmm1.i  = _mm_slli_epi64(xmm1.i, 0x34);
    xmm3.d  = _mm_mul_pd(xmm3.d, xmm4.d);
    xmm2.d  = _mm_add_pd(xmm6.d, *(__m128d*)d_k_180);
    xmm3.d  = _mm_add_pd(xmm3.d, *(__m128d*)d_k_190);
    xmm2.d  = _mm_mul_pd(xmm2.d, xmm6.d);
    xmm0.i  = _mm_or_si128(xmm0.i, *d_k_1A0);
    xmm7.d  = _mm_mul_pd(xmm7.d, xmm3.d);
    xmm2.d  = _mm_add_pd(xmm2.d, *(__m128d*)d_k_190);
    xmm0.d  = _mm_mul_pd(xmm0.d, xmm7.d);
    xmm6.i  = _mm_or_si128(*d_k_1A0, xmm1.i);
    xmm5.d  = _mm_mul_pd(xmm5.d, xmm2.d);
    xmm0.f  = _mm_cvtpd_ps(xmm0.d);
    xmm5.d  = _mm_mul_pd(xmm5.d, xmm6.d);
    xmm5.f  = _mm_cvtpd_ps(xmm5.d);
    xmm0.f  = _mm_shuffle_ps(xmm0.f, xmm5.f, _MM_SHUFFLE(1, 0, 1, 0));
    return xmm0.f;
}

static void log_a_test(benchmark::State& state) {
    float v = 0.0001f;
    for (auto _ : state) {
        __m128 val = _mm_set_ps(v, v + 0.01f, v + v, v + v * v);
        __m128 res = log_a(val);
        benchmark::DoNotOptimize(res);
        v += 0.0001f;
    }
}
BENCHMARK(log_a_test);

static void log_b_test(benchmark::State& state) {
    float v = 0.0001f;
    for (auto _ : state) {
        __m128 val = _mm_set_ps(v, v + 0.01f, v + v, v + v * v);
        __m128 res = log_b(val);
        benchmark::DoNotOptimize(res);
        v += 0.0001f;
    }
}
BENCHMARK(log_b_test);

static void pow_a_test(benchmark::State& state) {
    float v = 0.0001f;
    for (auto _ : state) {
        __m128 val = _mm_set_ps(v, v + 0.01f, v + v, v + v * v);
        __m128 res = pow_a(val);
        benchmark::DoNotOptimize(res);
        v += 0.0001f;
    }
}
BENCHMARK(pow_a_test);

static void pow_b_test(benchmark::State& state) {
    float v = 0.0001f;
    for (auto _ : state) {
        __m128 val = _mm_set_ps(v, v + 0.01f, v + v, v + v * v);
        __m128 res = pow_b(val);
        benchmark::DoNotOptimize(res);
        v += 0.0001f;
    }
}
BENCHMARK(pow_b_test);

BENCHMARK_MAIN();
