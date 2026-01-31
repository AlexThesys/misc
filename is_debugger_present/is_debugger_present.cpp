#include <stdio.h>
#include <stdint.h>

extern "C" {
    bool __is_debugger_present_beingDebugged();
    bool __is_debugger_present_NtGlobalFlag();
}

#include <windows.h>
#include <winternl.h>

//typedef enum _PROCESSINFOCLASS {
//    ProcessDebugPort = 7,
//    ProcessDebugObjectHandle = 30,
//    ProcessDebugFlags = 31
//} PROCESSINFOCLASS;

typedef NTSTATUS(NTAPI* NtQueryInformationProcess_t)(HANDLE, PROCESSINFOCLASS, PVOID, ULONG, PULONG);

bool is_debugger_present_NtQuery() {
    NtQueryInformationProcess_t NtQueryInformationProcess = (NtQueryInformationProcess_t)GetProcAddress(GetModuleHandleA("ntdll.dll"),"NtQueryInformationProcess");

    HANDLE debugPort = NULL;
    NTSTATUS status = NtQueryInformationProcess(GetCurrentProcess(), ProcessDebugPort, &debugPort, sizeof(debugPort), NULL);

    return NT_SUCCESS(status) && debugPort != NULL;
}

bool has_hw_breakpoints() {
    CONTEXT ctx = { 0 };
    ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;

    if (!GetThreadContext(GetCurrentThread(), &ctx))
        return false;

    return (ctx.Dr0 || ctx.Dr1 || ctx.Dr2 || ctx.Dr3);
}

int main() {
    int res = 0;
    if (__is_debugger_present_beingDebugged()) {
        puts("__is_debugger_present_beingDebugged");
        res++;
    }
    if (__is_debugger_present_NtGlobalFlag()) {
        puts("__is_debugger_present_NtGlobalFlag");
        res++;
    }
    if (is_debugger_present_NtQuery()) {
        puts("is_debugger_present_NtQuery");
        res++;
    }
    if (has_hw_breakpoints()) {
        puts("has_hw_breakpoints");
        res++;
    }

	return res;
}