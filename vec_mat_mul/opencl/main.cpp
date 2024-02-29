#define _CRT_SECURE_NO_WARNINGS

#include "benchmark/benchmark.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <random>

#include <immintrin.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE 0x100000

#define _min(x, y) ((x) < (y)) ? (x) : (y)
#define _max(x, y) ((x) > (y)) ? (x) : (y)

#define check_result(msg)							\
{													\
	if (ret != CL_SUCCESS) {						\
		printf("!%s, Error Code: %d", msg, ret);	\
		return				ret;					\
	}												\
}

inline uint64_t multiple_of_n(uint64_t val, uint64_t n) {
    return ((val - 1) | (n - 1)) + 1;
}

struct data {
    float* result;
    uint16_t* matrix;
    float* vector;
    int num_rows;
    int num_cols;

    data(int r, int c) : num_rows(r), num_cols(c) {
        result = (float*)malloc(r * sizeof(float));
        matrix = (uint16_t*)malloc(r * c * sizeof(uint16_t));
        vector = (float*)malloc(c * sizeof(float) + sizeof(__m256));
    }

    float* vector_aligned() { return (float*)multiple_of_n((uint64_t)vector, sizeof(__m256)); }
    ~data() {
        free(result);
        free(matrix);
        free(vector);
    }
};

struct apsp_cl {
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem objects[3];
    cl_device_id device_id = 0;

    cl_int move_memory_to_device(data* g_data);
    cl_int move_constant_memory_to_device(data* g_data);
    cl_int move_memory_to_host(data* g_data);

    cl_int move_mapped_memory_to_device(data* g_data);
    cl_int move_mapped_memory_to_host(data* g_data);
public:
    cl_int init();
    cl_int destroy();
    cl_int setup_and_run(data* g_data);
};


cl_int apsp_cl::init() {
    FILE* fp = fopen("kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    char* source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    constexpr cl_uint max_platforms = 4;
    cl_platform_id platform_ids[max_platforms];
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(max_platforms, platform_ids, &ret_num_platforms);
    check_result("SO calc: CL failed to retrive platform IDs!");
    for (int i = 0; i < ret_num_platforms; i++) {
        ret = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
        if (ret == CL_SUCCESS) {
            break;
        }
    }
    check_result("SO calc: CL failed to find appropriate device!");

    // Create an OpenCL context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    check_result("SO calc: CL failed to create context!");
    // Create a command queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);  // in-order execution
    check_result("SO calc: CL failed to create command queue!");

    // Create a program from the kernel source
    program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, (const size_t*)&source_size, &ret);
    check_result("SO calc: CL failed to create programm!");

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, "-Werror -cl-denorms-are-zero -cl-fast-relaxed-math", NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("CL failed to build programm!, Error Code: %d\n", ret);
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char* log = (char*)malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
        free(log);

        free(source_str);

        return ret;
    }
    //check_result("SO calc: CL failed to build programm!");

    // Create the OpenCL kernels
    kernel = clCreateKernel(program, "vec_mat_mul", &ret);
    check_result("SO calc: CL failed to create kernel #0!");

    free(source_str);

    return ret;
}

cl_int apsp_cl::destroy() {
    cl_int ret = 0;
    ret |= clFlush(command_queue);
    ret |= clFinish(command_queue);
    ret |= clReleaseKernel(kernel);
    ret |= clReleaseProgram(program);
    ret |= clReleaseCommandQueue(command_queue);
    ret |= clReleaseContext(context);
    return ret;
}

cl_int apsp_cl::setup_and_run(data* g_data) {
    cl_int ret = CL_SUCCESS;

    const int num_rows = g_data->num_rows;
    const int num_cols = g_data->num_cols;

    // Execute the OpenCL kernel
    const size_t local_item_size = 0x80;
    const size_t global_item_size = multiple_of_n(num_rows, local_item_size);

    // check device capabilities
    cl_uint num_dim = 0;
    size_t val[3];						// should not be more than the 3
    ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &num_dim, NULL);
    check_result("SO calc: error retrieving device info #0.");
    assert(num_dim >= 2);
    ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * num_dim, &val, NULL);
    check_result("SO calc: error retrieving device info #2.");
    assert(local_item_size < val[0]);

    // Set the arguments of the kernel
    ret |= clSetKernelArg(kernel, 0, sizeof(cl_int), (void*)&num_cols);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&num_rows);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&objects[0]);
    ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&objects[1]);
    ret |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&objects[2]);

    check_result("Errors setting up kernel argumets!");

    ret |= clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    check_result("Errors occured during kernel execution!");

    return ret;
}

cl_int apsp_cl::move_constant_memory_to_device(data* g_data) {
    cl_int ret;

    const int r = g_data->num_rows;
    const int c = g_data->num_cols;
    const int m_size_bytes = r * c * sizeof(uint16_t);
    const uint8_t pattern = 0;

    // Create memory buffers on the device for each vector 
    objects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY/* | CL_MEM_ALLOC_HOST_PTR*/, m_size_bytes, NULL, &ret);
    ret = clEnqueueWriteBuffer(command_queue, objects[1], CL_TRUE, 0, m_size_bytes, g_data->matrix, 0, NULL, NULL);

    return ret;
}

cl_int apsp_cl::move_memory_to_device(data* g_data) {
    cl_int ret;

    const int r = g_data->num_rows;
    const int c = g_data->num_cols;
    const uint8_t pattern = 0;

    // Create memory buffers on the device for each vector 
    objects[0] = clCreateBuffer(context, CL_MEM_WRITE_ONLY/* | CL_MEM_ALLOC_HOST_PTR*/, r * sizeof(float), NULL, &ret);
    objects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY/* | CL_MEM_ALLOC_HOST_PTR*/, c * sizeof(float), NULL, &ret);

    clEnqueueFillBuffer(command_queue, objects[0], &pattern, sizeof(pattern), 0, r * sizeof(float), 0, NULL, NULL);
    ret |= clEnqueueWriteBuffer(command_queue, objects[2], CL_TRUE, 0, c * sizeof(float), g_data->vector_aligned(), 0, NULL, NULL);

    return ret;
}

cl_int apsp_cl::move_memory_to_host(data* g_data) {
    cl_int ret;

    ret = clEnqueueReadBuffer(command_queue, objects[0], CL_TRUE, 0, g_data->num_rows * sizeof(float), g_data->result, 0, NULL, NULL);

    ret |= clReleaseMemObject(objects[0]);
    ret |= clReleaseMemObject(objects[1]);
    ret |= clReleaseMemObject(objects[2]);

    return ret;
}

//----------------------------------------------

typedef float fp32;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint16_t fp16;
typedef int32_t s32;
typedef uint32_t u32;
typedef uint64_t u64;

constexpr u32 c = 512;
constexpr u32 r = 1000;

template <typename T>
bool	fp_similar(T a, T b, T cmp = T(0.00001f)) { return abs(a - b) <= cmp; }

void generate_data(fp16* tensor, fp32* vector, u32 height, u32 width) {
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
        fp16* t_row = &tensor[w * height];
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

void vec_mat_mul(fp32* res, const fp16* tensor, const fp32* vector, u32 height, u32 width);

//int main() {
//    cl_int ret = CL_SUCCESS;
//
//    data c_data(r, c);
//    generate_data(c_data.matrix, c_data.vector_aligned(), c_data.num_cols, c_data.num_rows);
//    memset(c_data.result, 0, c_data.num_rows * sizeof(fp32));
//
//    apsp_cl context;
//    ret = context.init();
//    check_result("init");
//    ret = context.move_constant_memory_to_device(&c_data);
//    check_result("move_constant_memory_to_device");
//    ret = context.move_memory_to_device(&c_data);
//    check_result("move_memory_to_device");
//    ret = context.setup_and_run(&c_data);
//    check_result("setup_and_run");
//    ret = context.move_memory_to_host(&c_data);
//    check_result("move_memory_to_host");
//    ret = context.destroy();
//    check_result("destroy");
//
//    alignas(alignof(__m256)) fp32 result_b[r];
//    memset(result_b, 0, r * sizeof(fp32));
//    vec_mat_mul(result_b, c_data.matrix, c_data.vector_aligned(), c, r);
//
//    for (u32 i = 0; i < r; i++) {
//        if (!fp_similar(c_data.result[i], result_b[i], 0.001f)) {
//            printf("result_a[%d]: %.4f != result_b[%d]: %.4f\n", i, c_data.result[i], i, result_b[i]);
//        }
//    }
//
//    return 0;
//}

static constexpr u32 stride_avx = sizeof(__m256) / sizeof(fp32);

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

void vec_mat_mul(fp32* res, const fp16* tensor, const fp32* vector, u32 height, u32 width) {	// height == row, width = col
    const u32 rem = height & (stride_avx - 1);
    const u32 height_trunc = height - rem;
    const u32 rem_offset = height_trunc - (stride_avx - rem);
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
}

////////////////////////////////////////////////////////

static void vec_mat_mul(benchmark::State& state) {
    const u32 height = c;
    const u32 width = r;

    alignas(alignof(__m256)) fp32 vector[c];
    fp16* tensor = (fp16*)malloc(r * c * sizeof(fp16) + sizeof(__m256));
    generate_data(tensor, vector, c, r);
    alignas(alignof(__m256)) fp32 result_a[r];
    for (auto _ : state) {
        vec_mat_mul(result_a, tensor, vector, height, width);
        benchmark::DoNotOptimize(result_a);
    }
    free(tensor);
}

static void vec_mat_mul_cl(benchmark::State& state) {
    const u32 height = c;
    const u32 width = r;

    data c_data(r, c);
    generate_data(c_data.matrix, c_data.vector_aligned(), c_data.num_cols, c_data.num_rows);
    memset(c_data.result, 0, c_data.num_rows * sizeof(fp32));

    apsp_cl context;
    context.init();
    context.move_constant_memory_to_device(&c_data);
    context.move_memory_to_device(&c_data);
    for (auto _ : state) {
        //context.move_memory_to_device(&c_data);
        context.setup_and_run(&c_data);
       // context.move_memory_to_host(&c_data);
        benchmark::DoNotOptimize(c_data.result);
    }
    context.move_memory_to_host(&c_data);
    context.destroy();
}

BENCHMARK(vec_mat_mul);
BENCHMARK(vec_mat_mul_cl);

BENCHMARK_MAIN();
