section .data

ALIGN 32
_rem_mask_table:
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      0ffffffffH
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      0ffffffffH
        DD      0ffffffffH
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      0ffffffffH
        DD      0ffffffffH
        DD      0ffffffffH
        DD      00H
        DD      00H
        DD      00H
        DD      00H
        DD      0ffffffffH
        DD      0ffffffffH
        DD      0ffffffffH
        DD      0ffffffffH
        DD      00H
        DD      00H
        DD      00H
        DD      0ffffffffH
        DD      0ffffffffH
        DD      0ffffffffH
        DD      0ffffffffH
        DD      0ffffffffH
        DD      00H
        DD      00H
        DD      0ffffffffH
        DD      0ffffffffH
        DD      0ffffffffH
        DD      0ffffffffH
        DD      0ffffffffH
        DD      0ffffffffH
        DD      00H
        DD      0ffffffffH
        DD      0ffffffffH
        DD      0ffffffffH
        DD      0ffffffffH
        DD      0ffffffffH
        DD      0ffffffffH
        DD      0ffffffffH

section .text

; rdi -> rcx
; rsi -> rdx
; rdx -> r8
; rcx -> r9
; r8 -> [rsp+0x28] -> rdi
; r9 -> rsi

; rcx -- tensor
; rdx -- vector
; r8 -- result
; r9 -- width (rows)
; [rsp+0x28] -- height (cols)

global multiply_matrix_by_vector_fp16_avx
multiply_matrix_by_vector_fp16_avx:
        mov     qword [rsp+0x08], rbx
        mov     qword [rsp+0x10], rsi
        mov     qword [rsp+0x18], rdi		
		mov		edi, dword [rsp+0x28]		
        push    r15
        push    r14
        push    r13
        push    r12
        mov     esi, edi
        and     esi, 7
        lea     eax, [rdi - 8]
        shl     rsi, 5
        lea     r10, [rel _rem_mask_table]
        vmovaps ymm0, yword [rsi + r10]
        vmaskmovps ymm0, ymm0, yword [rdx + 4*rax]
        test    r9d, r9d
        je      .LBB1_15
        mov     esi, edi
        and     edi, -8
        je      .LBB1_5
        mov     r10d, edi
        mov     r9d, r9d
        dec     r10
        shr     r10, 3
        inc     r10
        mov     r11d, r10d
        and     r11d, 3
        and     r10, -4
        add     rsi, rsi
        mov     rbx, r11
        shl     rbx, 4
        xor     r14d, r14d
        jmp     .LBB1_3
    ALIGN 16
    .LBB1_14:
        vcvtph2ps ymm2, oword [rcx + 2*rax]
        vmulps  ymm2, ymm0, ymm2
        vaddps  ymm1, ymm1, ymm2
        vextractf128 xmm2, ymm1, 1
        vaddps  xmm1, xmm2, xmm1
        vmovshdup xmm2, xmm1
        vaddps  xmm1, xmm1, xmm2
        vshufpd xmm2, xmm1, xmm1, 1
        vaddss  xmm1, xmm1, xmm2
        vmovss  dword [r8 + 4*r14], xmm1
        inc     r14
        add     rcx, rsi
        cmp     r14, r9
        je      .LBB1_15
    .LBB1_3:
        vxorps  xmm1, xmm1, xmm1
        cmp     edi, 25
        jae     .LBB1_9
        xor     r15d, r15d
        jmp     .LBB1_11
    .LBB1_9:
        mov     r12, r10
        xor     r15d, r15d
    .LBB1_10:
        vcvtph2ps ymm2, oword [rcx + 2*r15]
        vmulps  ymm2, ymm2, yword [rdx + 4*r15]
        vcvtph2ps ymm3, oword [rcx + 2*r15 + 16]
        vaddps  ymm1, ymm1, ymm2
        vmulps  ymm2, ymm3, yword [rdx + 4*r15 + 32]
        vaddps  ymm1, ymm1, ymm2
        vcvtph2ps ymm2, oword [rcx + 2*r15 + 32]
        vmulps  ymm2, ymm2, yword [rdx + 4*r15 + 64]
        vcvtph2ps ymm3, oword [rcx + 2*r15 + 48]
        vmulps  ymm3, ymm3, yword [rdx + 4*r15 + 96]
        vaddps  ymm1, ymm1, ymm2
        vaddps  ymm1, ymm1, ymm3
        add     r15, 32
        add     r12, -4
        jne     .LBB1_10
    .LBB1_11:
        test    r11, r11
        je      .LBB1_14
        lea     r12, [rdx + 4*r15]
        lea     r15, [rcx + 2*r15]
        xor     r13d, r13d
    ALIGN 16
    .LBB1_13:
        vcvtph2ps ymm2, oword [r15 + r13]
        vmulps  ymm2, ymm2, yword [r12 + 2*r13]
        vaddps  ymm1, ymm1, ymm2
        add     r13, 16
        cmp     rbx, r13
        jne     .LBB1_13
        jmp     .LBB1_14
    .LBB1_5:
        mov     edx, r9d
        cmp     r9d, 1
        jne     .LBB1_16
        xor     r9d, r9d
        jmp     .LBB1_7
    .LBB1_16:
        mov     edi, edx
        and     edi, -2
        lea     r10, [rax + rax]
        lea     r11, [r10 + 2*rsi]
        shl     rsi, 2
        xor     r9d, r9d
        vxorps  xmm1, xmm1, xmm1
    ALIGN 16
    .LBB1_17:
        vcvtph2ps ymm2, oword [rcx + r10]
        vmulps  ymm2, ymm0, ymm2
        vaddps  ymm2, ymm2, ymm1
        vextractf128 xmm3, ymm2, 1
        vaddps  xmm2, xmm3, xmm2
        vmovshdup xmm3, xmm2
        vaddps  xmm2, xmm2, xmm3
        vshufpd xmm3, xmm2, xmm2, 1
        vaddss  xmm2, xmm2, xmm3
        vmovss  dword [r8 + 4*r9], xmm2
        vcvtph2ps ymm2, oword [rcx + r11]
        vmulps  ymm2, ymm0, ymm2
        vaddps  ymm2, ymm2, ymm1
        vextractf128 xmm3, ymm2, 1
        vaddps  xmm2, xmm3, xmm2
        vmovshdup xmm3, xmm2
        vaddps  xmm2, xmm2, xmm3
        vshufpd xmm3, xmm2, xmm2, 1
        vaddss  xmm2, xmm2, xmm3
        vmovss  dword [r8 + 4*r9 + 4], xmm2
        add     r9, 2
        add     rdx, rsi
        cmp     rdi, r9
        jne     .LBB1_17
    .LBB1_7:
        test    sil, 1
        je      .LBB1_15
        vcvtph2ps ymm1, oword [rcx + 2*rax]
        vmulps  ymm0, ymm0, ymm1
        vxorps  xmm1, xmm1, xmm1
        vaddps  ymm0, ymm0, ymm1
        vextractf128 xmm1, ymm0, 1
        vaddps  xmm0, xmm1, xmm0
        vmovshdup xmm1, xmm0
        vaddps  xmm0, xmm0, xmm1
        vshufpd xmm1, xmm0, xmm0, 1
        vaddss  xmm0, xmm0, xmm1
        vmovss  dword [r8 + 4*r9], xmm0
    .LBB1_15:
        pop     r12
        pop     r13
        pop     r14
        pop     r15
		mov     rbx, qword [rsp+8]
        mov     rsi, qword [rsp+16]
        mov     rdi, qword [rsp+24]
        vzeroupper
        ret
