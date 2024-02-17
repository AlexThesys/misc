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
        push    rbx
        mov     ebx, r8d ; copy height
        lea     r10, [rel _rem_mask_table] ; load rem mask table address
        and     r8d, 7 ; compute height reminder
        mov     r11d, ebx ; copy height
        mov     eax, r8d ; copy reminder
        sub     r11d, r8d ; compute truncated height
        shl     rax, 5 ; reminder mask table offset in bytes
        lea     r9, dword [rbx-8] ; compute height - 8
        vmovdqu ymm0, yword [rax+r10] ; load the reminder mask
        vmaskmovps ymm5, ymm0, yword [rdx+r9*4] ; maskload the reminder of the vector
        test    ecx, ecx ; test the width
        je      LN19 ; return if zero
        mov     r10d, ebx ; copy height
        mov     ecx, ecx ; code alignment
        add     r10, r10 ; height in bytes
        mov     ebx, r9d ; copy height - 8 
        vxorps  xmm4, xmm4, xmm4 ; set to zero
    ;ALIGN 16
    LL4:
        xor     r9, r9 ; zeroize height offset 
        vmovups ymm2, ymm4 ; zeroize accum
        test    r11d, r11d ; check inner loop condition
        je      LN6 ; if zero go to reminder computation
    ALIGN 16
    LL7:
        vcvtph2ps ymm0, oword [rsi+r9*2] ; load the tensor value
        vmulps  ymm1, ymm0, yword [rdx+r9*4] ; multiply tensor value by vector value
        add     r9, 8 ; increment height offset
        vaddps  ymm2, ymm1, ymm2 ; add the product to the accum
        cmp     r9d, r11d ; compare height offset to truncated height
        jb      LL7 ; if less then do another iteration
    LN6:
        vcvtph2ps ymm0, oword [rsi+rbx*2] ; load tensor reminder value 
        vmulps  ymm1, ymm0, ymm5
        vaddps  ymm2, ymm1, ymm2
        vextractf128 xmm3, ymm2, 1
        vaddps  xmm3, xmm3, xmm2
        add     rsi, r10 ; increment tensor pointer in bytes
        vmovaps xmm1, xmm3 ; 
        vshufps xmm1, xmm1, xmm3, 0xb1
        vmovaps xmm0, xmm1
        vaddps  xmm0, xmm0, xmm3
        vmovhlps xmm1, xmm1, xmm0
        vaddss  xmm0, xmm0, xmm1
        vmovss  dword [rdi], xmm0 ; copy value to the result
        add     rdi, 4 ; increment result pointer
        sub     rcx, 1 ; decrement the width 
        jne     LL4 ; do another iteration if not zero
    LN19:
        vzeroupper
        pop     rbx
        ret
