#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <memory.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdalign.h>
#include <pthread.h>
#include <assert.h>

#define MAX_THREADS 0x8
#define MAX_SIZE 0x10000000
#define REG_SIZE 0x10
#define BYTE_LIM 0x100
#define row_major(i, j) \
((i)*REG_SIZE+(j))
#define MIN(x,y) (x) < (y) ? (x) : (y)

size_t read_file(const char*, uint8_t**, size_t*);
size_t calc_alignment(void*, const uint8_t);
void print_stats(const uint32_t* stats, int num_chunks);
void brute_force_key(const uint8_t*, uint32_t*, const size_t);
void reduce(uint32_t* stats, size_t num_chunks);
static void* func(void*);

typedef struct bfk_params {
    const uint8_t* buf;
    uint32_t* stats;
    size_t size;
} bfk_params;

int main (int argc, char **argv)
{
    if (argc < 2) {
        puts("Provide the filename!");
        return -1;
    }
    int tolerance = 0;
    if (argc < 3) {
        puts("No tolerance value supplied. Zero tolerance will be used.");
    } else {
        tolerance = atoi(argv[2]);
    }
    uint8_t *buf = NULL;
    size_t buf_offset;
    const size_t size = read_file(argv[1], &buf, &buf_offset);
    const uint32_t num_simd_blocks = size / REG_SIZE;
    const uint32_t num_thread_blocks = num_simd_blocks / BYTE_LIM;
    const int num_chunks = num_threads ? num_threads*(BYTE_LIM-1) : MIN(0xFF, num_simd_blocks);
    if (num_chunks <= tolerance) {
        printf("Max tolerance value for this file is: %d\n", num_chunks - 2);
        return -1;
    }
    uint32_t *stats = malloc(REG_SIZE*(num_threads+1)*BYTE_LIM*sizeof(uint32_t)); // +1 is to account for alignment
    const size_t stats_offset = calc_alignment(stats, REG_SIZE);
    if (num_threads < 1) {
        brute_force_key((const uint8_t*)(buf+buf_offset), (uint32_t*)(stats+stats_offset), num_simd_blocks);
    } else {
        pthread_t threads[MAX_THREADS];
        bfk_params params[MAX_THREADS];
        int result_code = 0;
        for (int i = 0; i < num_threads; ++i) {
            const uint32_t block_offset = BYTE_LIM * i;
            params[i] = (bfk_params){(const uint8_t*)(buf+buf_offset+block_offset), (stats+stats_offset+block_offset*REG_SIZE), 0xFF};
            result_code = pthread_create(&threads[i], NULL, func, (void*)&params[i]);
            assert(!result_code);
        }
        for (int i = 0; i < num_threads; ++i) {
            result_code = pthread_join(threads[i], NULL);
            assert(!result_code);
        }
    }
    reduce((uint32_t*)(stats+stats_offset), num_threads);
    print_stats((uint32_t*)(stats+stats_offset), num_chunks - tolerance);
    free(buf);
    free(stats);
    return EXIT_SUCCESS;
}

uint64_t calc_alignment(void* buf, const uint8_t align)
{
    const size_t addr = (size_t)buf;
    return (addr & (align-0x1));
}

size_t read_file(const char *fname, uint8_t **buf, size_t *offset)
{
    FILE* file = fopen(fname, "rb");
    if (!file) {
        puts("Error opening file!");
       return -1; 
    }
    fseek(file, 0L, SEEK_END);    
    const size_t file_size = ftell(file);
    if (file_size < 0x11) {
        puts("File size to small for any meaningfull processing!");
        fclose(file);
        return -1;
    }
    rewind(file);
    if (file_size > MAX_SIZE) {
        puts("File exceeding maximum size!");
        fclose(file);
        return -1;
    }
    const size_t size = (file_size & (~(REG_SIZE-0x1))) + REG_SIZE; // make multiple of REG_SIZE
    if (!(*buf = calloc((size + (REG_SIZE-0x1)), 0x1))) {  // add margin to aling by REG_SIZE
        puts("Buffer allocation failed!");
        fclose(file);
        return -1;
    } 
    *offset = calc_alignment((void*)*buf, REG_SIZE);
    if (fread((void*)*(buf + *offset), 1, file_size, file) != file_size) {
        puts("Error reading file!");
        fclose(file);
        return -1;
    }
    fclose(file);
    return size;
}

void print_stats(const uint32_t* stats, int num_chunks)
{
    // printf("num blocks: %lu\n", num_blocks);
    printf("%d-byte xor encryption key stats:\n", REG_SIZE);
    for (uint16_t i = 0u; i < REG_SIZE; i++) {
        printf("For byte #%d possible char codes are:\t", i);
        for (uint16_t j = 0u; j < BYTE_LIM; j++) {
            if (stats[row_major(j,i)] >= num_chunks) {
                printf("%x ", j);
            }
           //printf("%x ", stats[row_major(j,i)]);
        }
        puts("");
    }
}

void brute_force_key(const uint8_t* buf, uint32_t* stats, const size_t size)
{
    alignas(REG_SIZE) uint8_t stats_chunk[0x100][REG_SIZE];   
    memset(stats_chunk, 0x0, REG_SIZE*BYTE_LIM);

    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;
    xmm0 = _mm_set1_epi8(0x1);  // increment
    xmm4 = _mm_set1_epi8(0x1f); // lower limit = 0x20-0x1
    xmm5 = _mm_set1_epi8(0x7f); // upper limit = 0x7e+0x1
    const size_t num_blocks = MIN(0xFF, size); // max value uint8_t can hold
    size_t buf_offset = 0x0;
    for (size_t i = 0; i < num_blocks; i++) {
        xmm3 = _mm_setzero_si128(); // accumulator
        xmm2 = _mm_load_si128((__m128i*)(buf+buf_offset)); // encrypted bytes
        buf_offset += REG_SIZE;
        for (uint16_t j = 0; j < 0x100; j++) {
            xmm1 = _mm_xor_si128(xmm2, xmm3);
            xmm3 = _mm_add_epi8(xmm3, xmm0);
            xmm6 = _mm_cmplt_epi8(xmm1, xmm5);
            xmm1 = _mm_and_si128(xmm1, xmm6);
            xmm6 = _mm_load_si128((__m128i*)&stats_chunk[j][0]);
            xmm1 = _mm_cmpgt_epi8(xmm1, xmm4);
            xmm1 = _mm_and_si128(xmm1, xmm0);
            xmm6 = _mm_add_epi8(xmm6, xmm1);
            _mm_store_si128((__m128i*)&stats_chunk[j][0], xmm6);
        }
    }
    // widen to dwords
    for (uint16_t i = 0; i < BYTE_LIM; i++)
        for (uint16_t j = 0; j < REG_SIZE; j++)
            stats[row_major(i,j)] = (uint32_t)stats_chunk[i][j];

}

static void* func(void* params) {
    bfk_params* p = (bfk_params*)params;
    brute_force_key(p->buf, p->stats, p->size);
}

void reduce(uint32_t* stats, size_t num_chunks) {
    const uint32_t num_iter = REG_SIZE / sizeof(uint32_t);
    __m128i accum, xmm0; 
    for (uint32_t i = 1; i < num_chunks; i++) {
        const uint32_t offset = i * BYTE_LIM * REG_SIZE;
        for (uint32_t j = 0; j < BYTE_LIM; j++) {
            for (uint32_t k = 0; k < num_iter; k++) {
                accum = _mm_load_si128((__m128i*)(&stats[j*REG_SIZE])+k);
                xmm0 = _mm_load_si128((__m128i*)(&stats[offset+j*REG_SIZE])+k);
                accum = _mm_add_epi32(accum, xmm0);
                _mm_store_si128((__m128i*)(&stats[j*REG_SIZE])+k, accum);   
            }
        }
    }
}
