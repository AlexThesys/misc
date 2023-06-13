#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdalign.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <errno.h>

#include <pthread.h>
#include <unistd.h>
#include <zlib.h>
#include <sys/mman.h>

#include <immintrin.h>

#define CHECK_INTERVAL_MS 500
#define CODE_BLOCK_SIZE 0x40
#define FILE_PATH_SIZE 0X400

enum elf_offsets {
    eo_ph = 0x20,
    eo_phsize = 0x36,
    eo_phnum = 0x38,
};

typedef struct code_info {
    uint32_t* code_hashes;
    uint8_t* mem_code_ptr;
    uint64_t file_code_offset;
    int num_code_blocks;
} code_info;

// FWD
static void* detect_bp_mt(void*);
static int compute_code_hashes(code_info* cinfo);
static void check_code_hashes(const code_info* cinfo);
static void remove_breakpoints(const code_info* cinfo, int block_id);
static uint64_t parse_memory_maps();

// global variables
static volatile int stop_watching;
static volatile int code_check_done; // might not be needed
static char file_path[FILE_PATH_SIZE];
static size_t page_size = 0x1000;
static pthread_t watchdog_thread;

void __attribute__ ((noinline)) some_test_code() {
    puts("...");
}

static void sleep_ms(int msec) {
    int res;
    struct timespec ts;

    ts.tv_sec = msec / 1000;
    ts.tv_nsec = (msec % 1000) * 1000000;

    do {
        res = nanosleep(&ts, &ts);
    } while (res && errno == EINTR);
}

void __attribute__ ((constructor)) pre_main(){
    int result;
    result = readlink("/proc/self/exe", file_path, FILE_PATH_SIZE);
    assert(-1 != result);
    page_size = sysconf(_SC_PAGE_SIZE);
    assert((size_t)-1 != page_size);
    result = pthread_create(&watchdog_thread, NULL, detect_bp_mt, NULL);
    assert(!result);
    while (!code_check_done)
        sleep_ms(CHECK_INTERVAL_MS);
}

int main() {
    int result;

    some_test_code();

    // press any key to stop the programm
    puts("press any key to stop the programm");
    getc(stdin);
    stop_watching = 1;

    result = pthread_join(watchdog_thread, NULL);
    assert(!result);

    return EXIT_SUCCESS;
}

static void* detect_bp_mt(void*) {
    int result;
    code_info cinfo = {NULL, NULL, 0, 0};

    result = compute_code_hashes(&cinfo);
    if (result == -1) {
        puts("Failed to compute code hashes! Thread exiting...");
        free(cinfo.code_hashes);
        return NULL;
    }
    // periodically check the code to detect and remove breakpoints
    while (!stop_watching) {
        check_code_hashes(&cinfo);

        sleep_ms(CHECK_INTERVAL_MS);
    }

    free(cinfo.code_hashes);

    return NULL;
}

typedef struct ph_hdr {
	uint32_t	p_type;
	uint32_t	p_flags;
	uint64_t    p_offset;
	uint64_t    p_vaddr;
	uint64_t    p_paddr;
	uint64_t    p_filesz;
	uint64_t    p_memsz;
	uint64_t    p_align;
} ph_hdr;

#define PT_LOAD 0x01
#define PF_X 	0x1 // 	Execute
#define PF_W 	0x2 // 	Write
#define PF_R 	0x4 // 	Read

#define CRC32_INIT 0xFFFFFFFF

static int compute_code_hashes(code_info* cinfo) {
    uint16_t word;
    int ph_num;
    ph_hdr hdr;
    // get address of the LOAD segment containing text section
    FILE* fd = fopen(file_path, "rb");
    if (!fd) {
        puts("Failed opening programm file!");
        return -1;
    }
    // parse file and program headers
    fseek(fd, eo_phsize, SEEK_SET);
    fread(&word, sizeof(word), 1, fd);
    assert((size_t)word == sizeof(hdr));

    fseek(fd, eo_phnum, SEEK_SET);
    fread(&word, sizeof(word), 1, fd);
    ph_num = (int)word;

    fseek(fd, eo_ph, SEEK_SET);
    fread(&word, sizeof(word), 1, fd);
    fseek(fd, word, SEEK_SET);

    do {
        fread(&hdr, sizeof(hdr), 1, fd);
        if ((hdr.p_type == PT_LOAD) && (hdr.p_flags == (PF_X | PF_R)))
            break;     
    } while (--ph_num > 0); 
    
    // compute code hashes from file
    assert(hdr.p_align >= CODE_BLOCK_SIZE);
    assert(hdr.p_align == page_size);
    cinfo->file_code_offset = hdr.p_offset;
    cinfo->num_code_blocks = (hdr.p_filesz + CODE_BLOCK_SIZE - 1) / (CODE_BLOCK_SIZE); // for now the tail is always rounded up to another block
    fseek(fd, hdr.p_offset, SEEK_SET); 
    cinfo->code_hashes = (uint32_t*)malloc(sizeof(uint32_t) * cinfo->num_code_blocks);
    for (int i = 0, sz = cinfo->num_code_blocks; i < sz; i++) {
        uint8_t code[CODE_BLOCK_SIZE];
        fread(code, CODE_BLOCK_SIZE, 1, fd);
        cinfo->code_hashes[i] = crc32(CRC32_INIT, code, CODE_BLOCK_SIZE); 
    }
    fclose(fd);

    // parse map file
    cinfo->mem_code_ptr = (uint8_t*)parse_memory_maps();

    return 0;
}

static void check_code_hashes(const code_info* cinfo) {
    const uint8_t *mem_code_ptr = cinfo->mem_code_ptr;
    for (int i = 0, sz = cinfo->num_code_blocks; i < sz; i++, mem_code_ptr+=CODE_BLOCK_SIZE) {
        const uint32_t crc_32 = crc32(CRC32_INIT, mem_code_ptr, CODE_BLOCK_SIZE);
        if (crc_32 != cinfo->code_hashes[i])
            remove_breakpoints(cinfo, i);
    }
    code_check_done = 1;
}

static void remove_breakpoints(const code_info* cinfo, int block_id) {
    int result;
    uint8_t buffer[CODE_BLOCK_SIZE];
    uint8_t *mem_code_ptr, *mem_code_page;
    
    const uint64_t block_offset = block_id * CODE_BLOCK_SIZE;
    mem_code_ptr = cinfo->mem_code_ptr + block_offset;
    const uint64_t file_code_offset = cinfo->file_code_offset + block_offset;

    FILE* fd = fopen(file_path, "rb");
    fseek(fd, file_code_offset, SEEK_SET);
    if (!fd) {
        puts("Failed opening programm file!");
        return;
    }
    fread(buffer, sizeof(buffer), 1, fd);
    fclose(fd);
    
    // make code page writable
    mem_code_page = (uint8_t*)((uintptr_t)mem_code_ptr & ~(page_size - 1));
    result = mprotect(mem_code_page, page_size, PROT_READ | PROT_WRITE | PROT_EXEC);
    assert(-1 != result);

    // remove the breakpoint(s) 
    // breakpoint is a single 0xCC byte - so no problems with going across the page boundaries
#ifndef NDEBUG
    for (int i = 0; i < CODE_BLOCK_SIZE; i++) {
        if (mem_code_ptr[i] != buffer[i])
            printf("Code bytes mismatch at: %p :\t %d %d\n", (mem_code_page+i), (int)mem_code_ptr[i], (int)buffer[i]);
    }
#endif // NDEBUG
    memcpy(mem_code_ptr, buffer, sizeof(buffer));
    _mm_sfence();

    // make code page unwritable
    result = mprotect(mem_code_page, page_size, PROT_READ | PROT_EXEC);
    assert(-1 != result);
}

static uint64_t parse_memory_maps() {
    alignas(alignof(int32_t)) const char perm[4] = "r-xp";
    char buffer[FILE_PATH_SIZE];
    FILE* fd;
    assert(sizeof(long long unsigned int) == sizeof(uint64_t));
    long long unsigned int start, end, offset;
    alignas(alignof(int32_t)) char flags[4];
    uint64_t result = (uint64_t)-1;

    // find starting address of the code in memory
    fd = fopen("/proc/self/maps", "rb");
    if (!fd) {
        puts("Failed opening maps file!");
        return -1;
    }

    // do parsing
    while (NULL != fgets(buffer, FILE_PATH_SIZE, fd)) {
        sscanf(buffer, "%llx-%llx %4c %llx", &start, &end, flags, &offset);
        // the first executable segment is the one we are looking for
        if (*(int32_t*)perm == *(int32_t*)flags) {
            result = (uint64_t)start;
            break;
        }
    }

    fclose(fd);

    assert(result != (uint64_t)-1);

    return result;
}
