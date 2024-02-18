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

; rdi -- result
; rsi -- tensor
; rdx -- vector
; rcx -- width
; r8 -- height

global __vec_mat_mul_linux64
__vec_mat_mul_linux64:
        push    r15
        push    r14
        push    r13
        push    r12
        push    rbx
        mov     r9d, r8d
        and     r9d, 7
        lea     eax, [r8 - 8]
        shl     r9, 5
        lea     r10, [rel _rem_mask_table]
        vmovaps ymm0, yword [r9 + r10]
        vmaskmovps ymm0, ymm0, yword [rdx + 4*rax]
        test    ecx, ecx
        je      .LBB0_15
        mov     r9d, r8d
        and     r8d, -8
        je      .LBB0_5
        mov     r10d, r8d
        mov     ecx, ecx
        dec     r10
        shr     r10, 3
        inc     r10
        mov     r11d, r10d
        and     r11d, 3
        and     r10, -4
        add     r9, r9
        mov     rbx, r11
        shl     rbx, 4
        xor     r14d, r14d
        jmp     .LBB0_3
    ALIGN 16
    .LBB0_14:
        vcvtph2ps ymm2, oword [rsi + 2*rax]
        vmulps  ymm2, ymm0, ymm2
        vaddps  ymm1, ymm1, ymm2
        vextractf128 xmm2, ymm1, 1
        vaddps  xmm1, xmm2, xmm1
        vmovshdup xmm2, xmm1
        vaddps  xmm1, xmm1, xmm2
        vshufpd xmm2, xmm1, xmm1, 1
        vaddss  xmm1, xmm1, xmm2
        vmovss  dword [rdi + 4*r14], xmm1
        inc     r14
        add     rsi, r9
        cmp     r14, rcx
        je      .LBB0_15
    .LBB0_3:
        vxorps  xmm1, xmm1, xmm1
        cmp     r8d, 25
        jae     .LBB0_9
        xor     r15d, r15d
        jmp     .LBB0_11
    .LBB0_9:
        mov     r12, r10
        xor     r15d, r15d
    .LBB0_10:
        vcvtph2ps ymm2, oword [rsi + 2*r15]
        vmulps  ymm2, ymm2, yword [rdx + 4*r15]
        vcvtph2ps ymm3, oword [rsi + 2*r15 + 16]
        vaddps  ymm1, ymm1, ymm2
        vmulps  ymm2, ymm3, yword [rdx + 4*r15 + 32]
        vaddps  ymm1, ymm1, ymm2
        vcvtph2ps ymm2, oword [rsi + 2*r15 + 32]
        vmulps  ymm2, ymm2, yword [rdx + 4*r15 + 64]
        vcvtph2ps ymm3, oword [rsi + 2*r15 + 48]
        vmulps  ymm3, ymm3, yword [rdx + 4*r15 + 96]
        vaddps  ymm1, ymm1, ymm2
        vaddps  ymm1, ymm1, ymm3
        add     r15, 32
        add     r12, -4
        jne     .LBB0_10
    .LBB0_11:
        test    r11, r11
        je      .LBB0_14
        lea     r12, [rdx + 4*r15]
        lea     r15, [rsi + 2*r15]
        xor     r13d, r13d
    ALIGN 16
    .LBB0_13:
        vcvtph2ps ymm2, oword [r15 + r13]
        vmulps  ymm2, ymm2, yword [r12 + 2*r13]
        vaddps  ymm1, ymm1, ymm2
        add     r13, 16
        cmp     rbx, r13
        jne     .LBB0_13
        jmp     .LBB0_14
    .LBB0_5:
        mov     edx, ecx
        cmp     ecx, 1
        jne     .LBB0_16
        xor     ecx, ecx
        jmp     .LBB0_7
    .LBB0_16:
        mov     r8d, edx
        and     r8d, -2
        lea     r10, [rax + rax]
        lea     r11, [r10 + 2*r9]
        shl     r9, 2
        xor     ecx, ecx
        vxorps  xmm1, xmm1, xmm1
    ALIGN 16
    .LBB0_17:
        vcvtph2ps ymm2, oword [rsi + r10]
        vmulps  ymm2, ymm0, ymm2
        vaddps  ymm2, ymm2, ymm1
        vextractf128 xmm3, ymm2, 1
        vaddps  xmm2, xmm3, xmm2
        vmovshdup xmm3, xmm2
        vaddps  xmm2, xmm2, xmm3
        vshufpd xmm3, xmm2, xmm2, 1
        vaddss  xmm2, xmm2, xmm3
        vmovss  dword [rdi + 4*rcx], xmm2
        vcvtph2ps ymm2, oword [rsi + r11]
        vmulps  ymm2, ymm0, ymm2
        vaddps  ymm2, ymm2, ymm1
        vextractf128    xmm3, ymm2, 1
        vaddps  xmm2, xmm3, xmm2
        vmovshdup xmm3, xmm2
        vaddps  xmm2, xmm2, xmm3
        vshufpd xmm3, xmm2, xmm2, 1
        vaddss  xmm2, xmm2, xmm3
        vmovss  dword [rdi + 4*rcx + 4], xmm2
        add     rcx, 2
        add     rsi, r9
        cmp     r8, rcx
        jne     .LBB0_17
    .LBB0_7:
        test    dl, 1
        je      .LBB0_15
        vcvtph2ps ymm1, oword [rsi + 2*rax]
        vmulps  ymm0, ymm0, ymm1
        vxorps  xmm1, xmm1, xmm1
        vaddps  ymm0, ymm0, ymm1
        vextractf128 xmm1, ymm0, 1
        vaddps  xmm0, xmm1, xmm0
        vmovshdup xmm1, xmm0
        vaddps  xmm0, xmm0, xmm1
        vshufpd xmm1, xmm0, xmm0, 1
        vaddss  xmm0, xmm0, xmm1
        vmovss  dword [rdi + 4*rcx], xmm0
    .LBB0_15:
        pop     rbx
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        vzeroupper
        ret
