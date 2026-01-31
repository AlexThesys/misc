section .text

global __is_debugger_present_beingDebugged
global __is_debugger_present_NtGlobalFlag

__is_debugger_present_beingDebugged:
		mov rax, gs: [0x60]    ; PEB
		mov al, [rax + 0x02]   ; BeingDebugged
		ret

__is_debugger_present_NtGlobalFlag:
		mov rax, gs: [0x60]    ; PEB
		mov al, [rax + 0xBC]   ; NtGlobalFlag
		ret