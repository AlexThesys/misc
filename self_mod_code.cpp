#include <stdio.h>
#include <stdint.h>
#include <windows.h>

const char* line_a = "Option A";
const char* line_b = "Option B";

__declspec(noinline) void bar() {
    puts(line_a);
}

int main() {
    puts("Calling bar...");
    bar();

    /* Actually we can pass the exact address of the start of the function as the first parameter 
    and the size of the function as the second. */
    char* bar_addr = (char*)bar;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    uint64_t page_size = si.dwPageSize;
    bar_addr = (char*)((uint64_t)bar_addr & ~(page_size - 1));
    DWORD dwOldProtect;
    if (VirtualProtect(bar_addr, page_size, PAGE_EXECUTE_READWRITE, &dwOldProtect) == -1) {
        puts("Error while changing page permissions of bar()");
    }
    unsigned int* instruction = (unsigned int*)((unsigned char*)bar + 0x03);
    const ptrdiff_t offset = line_b - line_a;
    *instruction += offset; // select address of the second string

    if (VirtualProtect(bar_addr, page_size, dwOldProtect, NULL) == -1) {
        puts("Error while tying to revert page permissions of bar()");
    }

    puts("Calling modified bar...");
    bar();
}

