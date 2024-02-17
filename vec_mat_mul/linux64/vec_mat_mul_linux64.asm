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
    ALIGN  32
__vec_mat_mul_linux64:
        mov     r9, rdx
        mov     edx, r8d
        lea     r11d, [r8-8]
        lea     r10, [rel _rem_mask_table]
        mov     rax, rdi
        and     edx, 7
        mov     edi, r8d
        sal     rdx, 5
        and     edi, -8
        vmovdqa ymm2, yword [r10+rdx]
        vmaskmovps ymm2, ymm2, yword [r9+r11*4]
        test    ecx, ecx
        je      .L12
        ;mov     r8d, r8d ; alignment
        mov     ecx, ecx
        lea     r10, [r8+r8]
        lea     rcx, [rax+rcx*4]
        mov     r8, rax
    ALIGN 16
    .L5:
        vxorps  xmm1, xmm1, xmm1
        test    edi, edi
        je      .L3
        xor     eax, eax
    ALIGN 16
    .L4:
        mov     edx, eax
        add     eax, 8
        vcvtph2ps ymm0, oword [rsi+rdx*2]
        vmulps  ymm0, ymm0, yword [r9+rdx*4]
        vaddps  ymm1, ymm1, ymm0
        cmp     eax, edi
        jb      .L4
    .L3:
        vcvtph2ps ymm0, oword [rsi+r11*2]
        vmulps  ymm0, ymm0, ymm2
        add     r8, 4
        add     rsi, r10
        vaddps  ymm0, ymm0, ymm1
        vextractf128 xmm1, ymm0, 0x01
        vaddps  xmm0, xmm0, xmm1
        vpermilps xmm1, xmm0, 0xb1
        vaddps  xmm0, xmm0, xmm1
        vmovhlps xmm1, xmm1, xmm0
        vaddss  xmm0, xmm0, xmm1
        vmovss  dword [r8-4], xmm0
        cmp     rcx, r8
        jne     .L5
    .L12:
        vzeroupper
        ret
