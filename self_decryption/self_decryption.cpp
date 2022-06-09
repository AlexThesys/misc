#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <vector>
#include <windows.h>
#include <xmmintrin.h>

#include "librandom.h"

#define NUM_FUNCS 2
#define NUM_CYCLES 16
#define MAGIC volatile uint64_t m = 0x88ff88ff88ff88ffL;\
_mm_sfence()

#define _max(x, y) ((x) > (y)) ? (x) : (y)

const char* line_a = "Text A";
const char* line_b = "Text B";

extern int get_keys(int argc, char** argv, int32_t* keys);

__declspec(noinline) void __cdecl foo() {
    // some code
    librandom::simple rnd_gen;
    rnd_gen.seed(0x7facacac);
    if (rnd_gen.i() & 0x1)
        puts(line_a);
    else
        puts(line_b);
}

__declspec(noinline) void __cdecl bar() {
    // some code
    librandom::simple rnd_gen;
    rnd_gen.seed(0x00bdbdbd);
    if (rnd_gen.i() & 0x1)
        puts(line_b);
    else
        puts(line_a);
}

volatile int64_t stop = 0;

static DWORD WINAPI check_dbg(LPVOID) {
    const DWORD sleep_tm = 1000;
    while (!stop) {
        if (IsDebuggerPresent()) {
            UINT exit_code = -1;
            ExitProcess(exit_code);
        }
        Sleep(sleep_tm);
    }
    return 0;
}

struct decrypt_data {
    void* f_ptr;
    uint64_t f_sz;
    void* params;
    uint8_t keys[NUM_CYCLES];
};

inline uint8_t cypher_func(int r, int k)
{
    return (uint8_t)((r * r) % (k + r));
}

void encrypt(decrypt_data* data)
{
    std::vector<uint8_t> temp(data->f_sz, 0);
    uint8_t* ptr = (uint8_t*)data->f_ptr;
    for (int i = 0; i < NUM_CYCLES; ++i) {
        for (int j = 0, len = (int)data->f_sz; j < len; j += 2) {
            temp[j / 2] = ptr[j];
            ptr[j] = ptr[j + 1];
            ptr[j + 1] = temp[j / 2] ^ cypher_func((int)ptr[j + 1], (int)data->keys[i]);
        }
    }
}

void decrypt(decrypt_data *data)
{
    std::vector<uint8_t> temp(data->f_sz, 0);
    uint8_t *ptr = (uint8_t*)data->f_ptr;
    for (int i = NUM_CYCLES - 1; i >= 0; --i) {
        for (int len = (int)data->f_sz, j = len - 1; j >= 0; j -= 2) {
            temp[(j - 1) / 2] = ptr[j];
            ptr[j] = ptr[j - 1];
            ptr[j - 1] = temp[(j - 1) / 2] ^ cypher_func((int)ptr[j - 1], (int)data->keys[i]);
        }
    }
}

ptrdiff_t get_func_length(void* f_ptr) {
    uint8_t* ptr = (uint8_t*)f_ptr;
    while (true) {
        if ((*ptr == 0xcc) && (*(ptr + 1) == 0xcc))
            break;
        ptr++;
    }
    return ptr - (uint8_t*)f_ptr;
}

template<typename func_t>
void decrypt_run_encrypt(decrypt_data* data, uint64_t page_size) {
    uint8_t* f_page_addr = (uint8_t*)data->f_ptr;
    f_page_addr = (uint8_t*)((uint64_t)f_page_addr & ~(page_size - 1));
    uint64_t fsize = data->f_sz + (uint64_t)((uint8_t*)data->f_ptr - f_page_addr);
    fsize = _max(fsize, page_size);
    DWORD dwOldProtect;

    //decyper
    VirtualProtect(f_page_addr, fsize, PAGE_EXECUTE_READWRITE, &dwOldProtect);
    decrypt(data);
    VirtualProtect(f_page_addr, fsize, dwOldProtect, NULL);

    // run
    func_t f = (func_t)data->f_ptr;
    f();

    //cypher
    VirtualProtect(f_page_addr, fsize, PAGE_EXECUTE_READWRITE, &dwOldProtect);
    encrypt(data);
    VirtualProtect(f_page_addr, fsize, dwOldProtect, NULL);
}

__declspec(noinline) void init_decrypt_data(decrypt_data* data, int32_t* seed) {
    MAGIC;
    data[0].f_ptr = (void*)foo;
    data[1].f_ptr = (void*)bar;
    data[0].f_sz = (uint64_t)get_func_length((void*)foo);
    data[1].f_sz = (uint64_t)get_func_length((void*)bar);
    librandom::simple rnd_gen;
    for (int i = 0; i < NUM_FUNCS; i++) {
        rnd_gen.seed(seed[i]);
        for (int j = 0; j < NUM_CYCLES; j++) {
            data[i].keys[j] = (uint8_t)rnd_gen.i(0xFF);
        }
    }
}

int main(int argc, char** argv) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    uint64_t page_size = si.dwPageSize;
    HANDLE t_handle = CreateThread(nullptr, page_size, check_dbg, nullptr, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    Sleep(500);

    int32_t seed[2];
    if (get_keys(argc, argv, seed))
        return 1;

    decrypt_data data[NUM_FUNCS];
    init_decrypt_data(data, seed);

    for (int i = 0; i < NUM_FUNCS; i++)
        decrypt_run_encrypt<void(*)()>(&data[i], page_size);

    stop = 1;
    if (t_handle) {
        WaitForSingleObject(t_handle, INFINITE);
        CloseHandle(t_handle);
    }

    return 0;
}