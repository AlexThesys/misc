#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <memory.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdalign.h>


#define MAX_SIZE 0x10000000
#define REG_SIZE 0x10
#define BYTE_LIM 0x8    // log2(0x100)
#define row_major(i, j) \
((i)*REG_SIZE+(j))
#define col_major(i, j) \
((j)<<BYTE_LIM+(i));   // j*0x100+i
#define MIN(x,y) (x) < (y) ? (x) : (y)

static const int bits2shift = (int)log2((double)REG_SIZE);

size_t read_file(const char*, uint8_t**, size_t*);
uint64_t calc_alignment(void*, int);
void print_stats(const uint8_t stats[][REG_SIZE], const size_t, int tolerance);
void brute_force_key(const uint8_t*, uint8_t stats[][REG_SIZE], const size_t);

int main(int argc, char** argv)
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
    uint8_t* buf = NULL;
    size_t offset;
    const size_t size = read_file(argv[1], &buf, &offset);
    if ((int)(MIN(0xff, (size >> bits2shift) - 1)) <= tolerance) {
        printf("Max tolerance value for this file is: %d\n", MIN(0xff, (size >> bits2shift) - 1) - 2);
        return -1;
    }
    // TODO: data type for stats should be at least uint32_t. For now the num_blocks can't be more than 0xff.
    alignas(REG_SIZE) uint8_t stats[0x100][REG_SIZE];
    memset(stats, 0x0, REG_SIZE << BYTE_LIM);
    brute_force_key((const uint8_t*)(buf + offset), stats, size);
    print_stats(stats, size, tolerance);
    free(buf);
    return EXIT_SUCCESS;
}

uint64_t calc_alignment(void* buf, int align)
{
    const size_t addr = (size_t)buf;
    return (addr & (align - 0x1));
}

size_t read_file(const char* fname, uint8_t** buf, size_t* offset)
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
    const size_t size = (file_size & (~(REG_SIZE - 0x1))) + REG_SIZE; // make multiple of REG_SIZE
    if (!(*buf = (uint8_t*)calloc((size + (REG_SIZE - 0x1)), 0x1))) {  // add margin to aling by REG_SIZE
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

void print_stats(const uint8_t stats[][REG_SIZE], const size_t size, int tolerance)
{
    //const size_t num_blocks = (size >> bits2shift) - 0x1; // the last block is padded so ignore it
    size_t num_blocks = MIN(0xff, (size >> bits2shift) - 1);
    num_blocks -= tolerance; // max value uint8_t can hold
    // printf("num blocks: %lu\n", num_blocks);
    printf("%d-byte xor encryption key stats:\n", REG_SIZE);
    for (uint16_t i = 0u; i < REG_SIZE; i++) {
        printf("For byte #%d possible char codes are:\t", i);
        for (uint16_t j = 0u; j < 0x100; j++) {
            if (stats[j][i] >= num_blocks) {
                printf("%x ", j);
            }
            //printf("%x ", stats[j][i]);
        }
        puts("");
    }
}

void brute_force_key(const uint8_t* buf, uint8_t stats[][REG_SIZE], const size_t size)
{
    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;
    xmm0 = _mm_set1_epi8(0x1);  // increment
    xmm4 = _mm_set1_epi8(0x1f); // lower limit = 0x20-0x1
    xmm5 = _mm_set1_epi8(0x7f); // upper limit = 0x7e+0x1
    //const size_t num_blocks = (size >> bits2shift) -1; // the last block is padded so ignore it
    const size_t num_blocks = MIN(0xff, (size >> bits2shift) - 1); // max value uint8_t can hold
    size_t buf_offset = 0x0;
    for (size_t i = 0; i < num_blocks; i++) {
        xmm3 = _mm_setzero_si128(); // accumulator
        xmm2 = _mm_load_si128((__m128i*)(buf + buf_offset)); // encrypted bytes
        buf_offset += REG_SIZE;
        for (uint16_t j = 0; j < 0x100; j++) {
            xmm1 = _mm_xor_si128(xmm2, xmm3);
            xmm3 = _mm_add_epi8(xmm3, xmm0);
            xmm6 = _mm_cmplt_epi8(xmm1, xmm5);
            xmm1 = _mm_and_si128(xmm1, xmm6);
            xmm6 = _mm_load_si128((__m128i*) & stats[j][0]);
            xmm1 = _mm_cmpgt_epi8(xmm1, xmm4);
            xmm1 = _mm_and_si128(xmm1, xmm0);
            xmm6 = _mm_add_epi8(xmm6, xmm1);
            _mm_store_si128((__m128i*) & stats[j][0], xmm6);
        }
    }
}
