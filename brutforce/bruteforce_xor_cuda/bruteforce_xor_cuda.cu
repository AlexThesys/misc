#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <memory.h>

#define MAX_BLOCKS 0x100
#define MAX_FSIZE 0x10000000
#define BYTE_LIM 0x100

#define MIN(x,y) (x) < (y) ? (x) : (y)

static int parse_args(int argc, char** argv);
static int read_file(const char*, uint8_t**, int);
void print_stats(const uint32_t* stats, int num_threads, int num_blocks);

static uint32_t flp2(uint32_t x) {
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x - (x >> 1);
}

static cudaError_t bruteforce(uint8_t* buf, uint32_t* stats, int num_threads, int num_chunks);

__global__ void bruteforce_kernel(uint8_t* buf, uint32_t* g_stats, int num_chunks) {
    uint32_t local_stats[BYTE_LIM] = { 0 };
    constexpr int32_t lower_lim = 0x20; // ascii space
    constexpr int32_t upper_lim = 0x7e; // ascii ~
    int curr_chunk = blockIdx.x;
    while (curr_chunk < num_chunks) {
        const int32_t ch = (int32_t)buf[threadIdx.x + curr_chunk * blockDim.x];
        for (int32_t j = 0; j < BYTE_LIM; j++) {
            const int32_t decrypted = ch ^ j;
            const int32_t greater = decrypted - lower_lim;
            const int32_t less = upper_lim - decrypted;
            const uint32_t inside = (~((uint32_t)greater | (uint32_t)less)) >> 31;
            local_stats[j] += inside;
        }
        curr_chunk += gridDim.x;
    }
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = tid * BYTE_LIM;
    for (int i = 0; i < BYTE_LIM; i++)
        g_stats[i + offset] = local_stats[i];
}

__global__ void reduction_kernel(uint32_t* g_stats) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = tid * BYTE_LIM;
    const int tid_2 = (gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    const int offset_2 = tid_2 * BYTE_LIM;
    for (int i = 0; i < BYTE_LIM; i++)
        g_stats[i + offset] += g_stats[i + offset_2];
}

int main(int argc, char** argv) {
    const int num_threads = parse_args(argc, argv);
    if (num_threads <= 0)
        return -1;

    uint8_t* buf = nullptr;

    const int num_chunks = read_file(argv[1], &buf, num_threads);
    if (num_chunks <= 0) {
        return -1;
    }

    int tolerance = 0;

    if (argc < 4) {
        puts("No tolerance value supplied. Zero tolerance will be used.");
    } else {
        tolerance = atoi(argv[3]);
        printf("Tolerance = %d chars.\n", tolerance);
    }
    if (num_chunks <= tolerance) {
        printf("Max tolerance value for this file is: %d\n", num_chunks - 2);
        return -1;
    }

    // stats[key_id][char+id]
    uint32_t* stats = (uint32_t*)malloc((size_t)(num_threads * BYTE_LIM * sizeof(uint32_t)));
    if (stats == nullptr) {
        puts("Error allocationg stats buffer!");
        free(buf);
        return -1;
    }

    cudaError_t cudaStatus = bruteforce(buf, stats, num_threads, num_chunks);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        free(buf);
        free(stats);
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        free(buf);
        free(stats);
        return 1;
    }

    print_stats(stats, num_threads, num_chunks - tolerance);

    free(buf);
    free(stats);
    return 0;
}

static cudaError_t bruteforce(uint8_t* buf, uint32_t* stats, int num_threads, int num_chunks)
{
    uint8_t* dev_buf = nullptr;
    uint32_t* dev_stats = nullptr;
    cudaError_t cudaStatus;

    cudaDeviceProp prop;
    int dev;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    cudaStatus = cudaChooseDevice(&dev, &prop);
    // Choose which GPU to run on, change this on a multi-GPU system.
    //cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    cudaGetDeviceProperties(&prop, dev);
    const int max_blocks = MIN(flp2(prop.multiProcessorCount * 2 * prop.maxBlocksPerMultiProcessor), MAX_BLOCKS);   // MAX_BLOCKS is experimental value

    int num_blocks = MIN(num_chunks, max_blocks);
    const size_t size = num_threads * num_chunks;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    cudaStatus = cudaMalloc((void**)&dev_buf, size * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_stats, (size_t)(num_blocks * num_threads) * BYTE_LIM * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_buf, buf, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    bruteforce_kernel << <num_blocks, num_threads >> > (dev_buf, dev_stats, num_chunks);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "bruteforce_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // reduction stage
    num_blocks /= 2;
    while (num_blocks) {
        reduction_kernel << <num_blocks, num_threads >> > (dev_stats);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "reduction_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        num_blocks /= 2;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(stats, dev_stats, (size_t)(num_threads * BYTE_LIM * sizeof(uint32_t)), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to calculate: %3.1f ms\n", elapsedTime);

Error:
    cudaFree(dev_buf);
    cudaFree(dev_stats);

    return cudaStatus;
}

static int parse_args(int argc, char** argv) {
    if (argc < 3) {
        puts("Provide the filename and key size (either 32, 64, 128 or 256 bytes)!");
        return -1;
    }
    int num_t = atoi(argv[2]);
    if ((num_t != 32) && (num_t != 64) && (num_t != 128) && (num_t != 256)) {
        puts("Key size must be either 32, 64, 128 or 256 bytes!");
        num_t = -1;
    }
    return num_t;
}

static int read_file(const char* fname, uint8_t** buf, int num_threads)
{
    FILE* file = fopen(fname, "rb");
    if (!file) {
        puts("Error opening file!");
        return -1;
    }
    fseek(file, 0L, SEEK_END);
    const size_t file_size = ftell(file);
    if (file_size < num_threads) {
        puts("File size to small for any meaningfull processing!");
        fclose(file);
        return -1;
    }
    rewind(file);
    if (file_size > MAX_FSIZE) {
        puts("File exceeding maximum size!");
        fclose(file);
        return -1;
    }
    const int num_chunks = (int)(flp2((uint32_t)(file_size / num_threads)));  // num_blocks should be power of two for the reduction step
    const int read_size = num_chunks * num_threads;
    if (!(*buf = (uint8_t*)malloc(read_size))) {
        puts("Buffer allocation failed!");
        fclose(file);
        return -1;
    }
    if (fread((void*)*(buf), 1, read_size, file) != read_size) {
        puts("Error reading file!");
        free(*buf);
        fclose(file);
        return -1;
    }
    fclose(file);

    return num_chunks;
}

void print_stats(const uint32_t* stats, int num_threads, int num_blocks)
{
    printf("%d-byte xor encryption key stats:\n", num_threads);
    for (uint16_t i = 0u; i < num_threads; i++) {
        printf("For byte #%d possible char codes are:\t", i);
        for (uint16_t j = 0u; j < BYTE_LIM; j++) {
            if (stats[i * BYTE_LIM + j] >= num_blocks) {
                printf("%x ", j);
            }
            //printf("%d ", stats[i * BYTE_LIM + j]);
        }
        puts("");
    }
}
