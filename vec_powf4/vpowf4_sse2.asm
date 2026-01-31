section .data

ALIGN 16
_s_data:
		DD 		07f800000h
		DD 		07f800000h
		DD 		07f800000h
		DD 		07f800000h
		DD 		0007fffffh
		DD 		0007fffffh
		DD 		0007fffffh
		DD 		0007fffffh
		DD 		07f7fffffh
		DD 		07f7fffffh
		DD 		07f7fffffh
		DD 		07f7fffffh
		DD 		00000007fh
		DD 		00000007fh
		DD 		00000007fh
		DD 		00000007fh
		DD 		0007fffffh
		DD 		0007fffffh
		DD 		0007fffffh
		DD 		0007fffffh
		DD 		0004afb10h
		DD 		0004afb10h
		DD 		0004afb10h
		DD 		0004afb10h
		DD 		000800000h
		DD 		000800000h
		DD 		000800000h
		DD 		000800000h
		DD 		000000000h
		DD 		03c080000h
		DD 		000000000h
		DD 		03c080000h
		DD 		03f000000h
		DD 		03f000000h
		DD 		03f000000h
		DD 		03f000000h
		DD 		07fffffffh
		DD 		07fffffffh
		DD 		07fffffffh
		DD 		07fffffffh
		DD 		03f800000h
		DD 		03f800000h
		DD 		03f800000h
		DD 		03f800000h
		DD 		000000000h
		DD 		040000000h
		DD 		000000000h
		DD 		040000000h
		DD 		048eb27cbh
		DD 		03fe55555h
		DD 		048eb27cbh
		DD 		03fe55555h
		DD 		0e1914edah
		DD 		03fd999b0h
		DD 		0e1914edah
		DD 		03fd999b0h
		DD 		0c344345dh
		DD 		03fd24185h
		DD 		0c344345dh
		DD 		03fd24185h
		DD 		03451a0f5h
		DD 		03fce772ah
		DD 		03451a0f5h
		DD 		03fce772ah
		DD 		0652b82feh
		DD 		03ff71547h
		DD 		0652b82feh
		DD 		03ff71547h
		DD 		000000000h
		DD 		043380000h
		DD 		000000000h
		DD 		043380000h
		DD 		040862000h
		DD 		040862000h
		DD 		040862000h
		DD 		040862000h
		DD 		0000003f2h
		DD 		000000000h
		DD 		0000003f2h
		DD 		000000000h
		DD 		0213bde9fh
		DD 		0401b6eb4h
		DD 		0213bde9fh
		DD 		0401b6eb4h
		DD 		0348d1e8dh
		DD 		04010bf17h
		DD 		0348d1e8dh
		DD 		04010bf17h
		DD 		050304bbeh
		DD 		0402a6a2ah
		DD 		050304bbeh
		DD 		0402a6a2ah
		DD 		0a22d3a94h
		DD 		04030a032h
		DD 		0a22d3a94h
		DD 		04030a032h
		DD 		0bb3fa39ch
		DD 		0c002807eh
		DD 		0bb3fa39ch
		DD 		0c002807eh
		DD 		0293ab434h
		DD 		0403dad2eh
		DD 		0293ab434h
		DD 		0403dad2eh
		DD 		0371366f6h
		DD 		000041d33h
		DD 		0371366f6h
		DD 		000041d33h
		DD 		0ffffffffh
		DD 		0ffffffffh
		DD 		0ffffffffh
		DD 		0ffffffffh

section .text

; agruments:
; xmm0 -- mantissa
; xmm1 -- exponent

global u_vpowf4_sse2
ALIGN 32
u_vpowf4_sse2:
		push        rbx  
		push        rsi  
		push        rdi  
		push        r12  
		push        r13  
		push        r14  
		push        r15  
		sub         rsp,0F0h  
		movaps      oword [rsp+0C0h],xmm6  
		movaps      oword [rsp+0B0h],xmm7  
		pxor        xmm6,xmm6  
		movdqa      xmm7,oword [rel _s_data]  
		movdqa      xmm2,oword [rel _s_data+50h]  
		movdqa      xmm3,xmm1  
		movdqa      xmm4,oword [rel _s_data+40h]  
		movdqa      xmm5,oword [rel _s_data+0A0h]  
		movdqa      oword [rsp+0D0h],xmm1  
		pand        xmm7,xmm1  
		pcmpgtd     xmm1,xmm6  
		pcmpgtd     xmm3,oword [rel _s_data+20h]  
		cmpnltps    xmm7,oword [rel _s_data]  
		pand        xmm4,xmm0  
		pcmpeqd     xmm6,xmm0  
		pandn       xmm3,xmm1  
		paddd       xmm2,xmm4  
		movdqa      xmm1,xmm5  
		pand        xmm6,xmm3  
		movdqa      xmm3,oword [rel _s_data+10h]  
		pand        xmm2,oword [rel _s_data+60h]  
		pcmpgtd     xmm3,xmm0  
		pmovmskb    edx,xmm6  
		pxor        xmm6,oword [rel _s_data+1B0h]  
		pxor        xmm1,xmm2  
		psrld       xmm2,17h  
		por         xmm4,xmm1  
		movdqa      xmm1,xmm0  
		pmovmskb    eax,xmm7  
		pxor        xmm7,xmm7  
		movdqa      oword [rsp+20h],xmm6  
		pand        xmm6,oword [rel _s_data+0A0h]  
		xor         edx,0FFFFh  
		je          .L1
		pslld       xmm1,1  
		movdqa      oword [rsp+0E0h],xmm0  
		pcmpgtd     xmm0,oword [rel _s_data+20h]  
		psrld       xmm1,18h  
		addps       xmm5,xmm4  
		psubd       xmm1,oword [rel _s_data+30h]
		movdqa      oword [rsp+30h],xmm6  
		pxor        xmm6,xmm6  
		pand        xmm1,oword [rsp+20h]  
		por         xmm0,xmm3  
		unpcklps    xmm6,xmm4  
		paddd       xmm1,xmm2  
		pmovmskb    ecx,xmm0  
		and         ecx,edx  
		jne         .L2
	.L7:
		rcpps       xmm5,xmm5  
		unpckhps    xmm7,xmm4  
		movdqa      xmm2,oword [rel _s_data+70h]  
		cvtps2pd    xmm0,xmm5  
		psrlq       xmm6,4  
		movapd      xmm3,oword [rel _s_data+0B0h]  
		psrlq       xmm7,4  
		paddd       xmm6,xmm2  
		shufps      xmm5,xmm5,0EEh  
		paddd       xmm7,xmm2  
		movapd      xmm4,xmm3  
		movapd      xmm2,xmm6  
		mulpd       xmm6,xmm0  
		subpd       xmm3,xmm2  
		subpd       xmm6,xmm4  
		cvtps2pd    xmm5,xmm5  
		subpd       xmm4,xmm7  
		mulpd       xmm0,xmm6  
		movapd      xmm6,xmm7  
		mulpd       xmm2,xmm0  
		mulpd       xmm3,xmm0  
		mulpd       xmm7,xmm5  
		addpd       xmm2,oword [rel _s_data+0B0h]  
		mulpd       xmm2,xmm3  
		subpd       xmm7,oword [rel _s_data+0B0h]  
		movapd      xmm3,xmm2  
		mulpd       xmm7,xmm5  
		movapd      xmm0,oword [rel _s_data+0F0h]  
		mulpd       xmm2,xmm2  
		mulpd       xmm6,xmm7  
		mulpd       xmm4,xmm7  
		addpd       xmm6,oword [rel _s_data+0B0h]  
		movapd      xmm5,xmm0  
		mulpd       xmm6,xmm4  
		movdqa      xmm7,xmm1  
		mulpd       xmm0,xmm2  
		movapd      xmm4,xmm6  
		cvtdq2pd    xmm1,xmm1  
		mulpd       xmm6,xmm6  
		addpd       xmm0,oword [rel _s_data+0E0h]
		pshufd      xmm7,xmm7,0EEh  
		mulpd       xmm5,xmm6  
		mulpd       xmm0,xmm2  
		addpd       xmm5,oword [rel _s_data+0E0h]  
		mulpd       xmm3,oword [rel _s_data+100h]  
		addpd       xmm0,oword [rel _s_data+0D0h]  
		cvtdq2pd    xmm7,xmm7  
		mulpd       xmm5,xmm6  
		mulpd       xmm0,xmm2  
		mulpd       xmm4,oword [rel _s_data+100h]  
		addpd       xmm5,oword [rel _s_data+0D0h]  
		addpd       xmm0,oword [rel _s_data+0C0h]  
		mulpd       xmm5,xmm6  
		mulpd       xmm0,xmm2  
		cvtps2pd    xmm2,qword [rsp+0D0h]  
		addpd       xmm5,oword [rel _s_data+0C0h]  
		addpd       xmm0,oword [rel _s_data+0B0h]  
		mulpd       xmm5,xmm6  
		cvtps2pd    xmm6,qword [rsp+0D8h]  
		mulpd       xmm3,xmm0  
		addpd       xmm5,oword [rel _s_data+0B0h]  
		addpd       xmm3,xmm1  
		mulpd       xmm5,xmm4  
		mulpd       xmm3,xmm2  
		addpd       xmm5,xmm7  
		mulpd       xmm5,xmm6  
		movapd      oword [rsp+90h],xmm3  
		movapd      oword [rsp+0A0h],xmm5  
		or          eax,ecx  
		jne         .L3
	.L35: 
		movapd      xmm0,oword [rel _s_data+110h]
		movaps      xmm7,xmm3  
		movapd      xmm4,xmm0  
		movdqa      xmm2,oword [rel _s_data+130h]  
		addpd       xmm0,xmm3  
		movapd      xmm1,xmm4  
		subpd       xmm4,xmm0  
		shufps      xmm7,xmm5,0DDh  
		addpd       xmm3,xmm4  
		pand        xmm7,oword [rel _s_data+90h]  
		movapd      xmm6,xmm1  
		pcmpgtd     xmm7,oword [rel _s_data+120h]  
		paddd       xmm0,xmm2  
		pmovmskb    eax,xmm7  
		addpd       xmm1,xmm5  
		movapd      xmm7,oword [rel _s_data+140h]  
		subpd       xmm6,xmm1  
		movapd      xmm4,oword [rel _s_data+150h]  
		addpd       xmm5,xmm6  
		addpd       xmm7,xmm3  
		paddd       xmm1,xmm2  
		movapd      xmm6,xmm5  
		mulpd       xmm7,xmm3  
		addpd       xmm5,xmm4  
		addpd       xmm4,xmm3  
		addpd       xmm7,oword [rel _s_data+160h]  
		movapd      xmm2,xmm6  
		mulpd       xmm5,xmm6  
		mulpd       xmm4,xmm3  
		addpd       xmm2,oword [rel _s_data+140h]  
		psllq       xmm0,34h  
		addpd       xmm5,oword [rel _s_data+170h]  
		addpd       xmm4,oword [rel _s_data+170h]  
		mulpd       xmm2,xmm6  
		mulpd       xmm7,xmm4  
		addpd       xmm2,oword [rel _s_data+160h]  
		movapd      xmm4,xmm3  
		mulpd       xmm5,xmm2  
		addpd       xmm3,oword [rel _s_data+180h]  
		psllq       xmm1,34h  
		movapd      xmm2,oword [rel _s_data+180h]  
		mulpd       xmm3,xmm4  
		addpd       xmm2,xmm6  
		addpd       xmm3,oword [rel _s_data+190h]  
		mulpd       xmm2,xmm6  
		por         xmm0,oword [rel _s_data+1A0h]  
		mulpd       xmm7,xmm3  
		addpd       xmm2,oword [rel _s_data+190h]  
		movdqa      xmm6,oword [rel _s_data+1A0h]  
		mulpd       xmm0,xmm7  
		por         xmm6,xmm1  
		mulpd       xmm5,xmm2  
		cvtpd2ps    xmm0,xmm0  
		mulpd       xmm5,xmm6  
		cvtpd2ps    xmm5,xmm5  
		shufps      xmm0,xmm5,44h  
		pand        xmm0,oword [rsp+20h]  
		test        eax,eax  
		jne         .L4
		jmp         .L5
	.L4:  
		movaps      xmm4,oword [rsp+90h]  
		movaps      xmm5,oword [rsp+0A0h]  
		movdqa      xmm7,oword [rel _s_data+1B0h]  
		movdqa      xmm1,oword [rel _s_data+120h]  
		movaps      xmm2,xmm4  
		shufps      xmm4,xmm5,0DDh  
		pxor        xmm6,xmm6  
		pcmpeqd     xmm7,xmm4  
		pcmpgtd     xmm6,xmm4  
		pand        xmm4,oword [rel _s_data+90h]  
		shufps      xmm2,xmm5,88h  
		pcmpgtd     xmm4,xmm1  
		movdqa      xmm3,xmm7  
		pxor        xmm7,xmm4  
		pandn       xmm4,xmm0  
		pandn       xmm6,xmm7  
		movdqa      xmm0,oword [rel _s_data]  
		pand        xmm2,xmm3  
		pand        xmm0,xmm6  
		por         xmm2,xmm4  
		por         xmm0,xmm2  
		pand        xmm0,oword [rsp+20h]  
		jmp         .L5
	.L2:  
		movaps      xmm7,oword [rsp+20h]  
		movaps      xmm3,oword [rsp+20h]  
		movaps      xmm2,oword [rsp+20h]  
		pand        xmm3,oword [rsp+0D0h]  
		pandn       xmm2,oword [rel _s_data+0A0h]  
		pand        xmm7,oword [rsp+0E0h]  
		por         xmm3,xmm2  
		por         xmm2,xmm7  
		movaps      xmm0,xmm2  
		movaps      oword [rsp+0D0h],xmm3  
		movdqa      xmm3,oword [rel _s_data+10h]  
		pcmpgtd     xmm0,oword [rel _s_data+20h]  
		pcmpgtd     xmm3,xmm2  
		movaps      oword [rsp+0E0h],xmm2  
		por         xmm0,xmm3  
		movapd      xmm7,oword [rel _s_data+10h]  
		pand        xmm2,oword [rel _s_data+90h]  
		pcmpgtd     xmm7,xmm2  
		movaps      xmm2,xmm7  
		pxor        xmm7,xmm7  
		pand        xmm3,xmm2  
		pmovmskb    ecx,xmm3  
		test        ecx,ecx  
		je          .L6
		movaps      xmm6,oword [rsp+0E0h]  
		pand        xmm6,oword [rel _s_data+90h]  
		movaps      xmm1,oword [rel _s_data+0A0h]  
		movaps      xmm5,xmm1  
		pxor        xmm0,xmm3  
		movaps      xmm2,oword [rel _s_data+50h]  
		pand        xmm1,xmm3  
		por         xmm6,xmm1  
		pand        xmm3,oword [rel _s_data+80h]  
		subps       xmm6,xmm1  
		paddd       xmm3,xmm5  
		movdqa      xmm4,xmm6  
		movaps      xmm1,xmm5  
		pand        xmm4,oword [rel _s_data+40h]  
		psrld       xmm3,17h  
		paddd       xmm2,xmm4  
		pslld       xmm6,1  
		pand        xmm2,oword [rel _s_data+60h]  
		pxor        xmm1,xmm2  
		psrld       xmm2,17h  
		por         xmm4,xmm1  
		movdqa      xmm0,oword [rsp+0E0h]  
		movdqa      xmm1,oword [rsp+0E0h]  
		pcmpgtd     xmm0,xmm7  
		pcmpgtd     xmm1,oword [rel _s_data+20h]  
		pxor        xmm0,oword [rel _s_data+1B0h]  
		por         xmm0,xmm1  
		pmovmskb    ecx,xmm0  
		movdqa      xmm1,xmm6  
		psrld       xmm1,18h  
		pxor        xmm6,xmm6  
		addps       xmm5,xmm4  
		psubd       xmm1,xmm3  
		unpcklps    xmm6,xmm4  
		paddd       xmm1,xmm2  
		jmp         .L7
	.L6:  
		pxor        xmm6,xmm6  
		pmovmskb    ecx,xmm0  
		unpcklps    xmm6,xmm4  
		jmp         .L7
	.L3:  
		movapd      oword [rsp+90h],xmm3  
		movapd      oword [rsp+0A0h],xmm5  
		mov         dword [rsp+88h],ecx  
		mov         r12d,dword [rsp+88h]  
		lea         rdi,[rsp+90h]  
		xor         esi,esi
	.L34:  
		mov         r13d,dword [rsp+rsi*4+0E0h]  
		mov         r14d,r13d  
		mov         ebx,dword [rsp+rsi*4+0D0h]  
		mov         r15d,ebx  
		and         r14d,7FFFFFFFh  
		and         r15d,7FFFFFFFh  
		jne         .L8 
		mov         dword [rdi+4],0FFFFFFFFh  
		mov         dword [rdi],3F800000h  
		jmp         .L9
	.L8:  
		cmp         r14d,7F800000h  
		ja          .L36  
		cmp         r15d,7F800000h  
		jbe         .L37
	.L36:
		mov         eax,7FFFFFFFh  
		cmp         r13d,3F800000h  
		mov         dword [rdi+4],0FFFFFFFFh  
		cmove       eax,r13d  
		mov         dword [rdi],eax  
		jmp         .L9
	.L37: 		
		test        r12d,1  
		je          .L10  
		test        r14d,r14d  
		je          .L11  
		cmp         r14d,7F800000h  
		je          .L12  
		mov         dword [rsp+80h],ebx  
		mov         edx,dword [rsp+80h]  
		pxor        xmm0,xmm0  
		mov         eax,edx  
		and         edx,7F800000h  
		and         eax,7FFFFFFFh  
		cmp         edx,4A800000h  
		ja          .L13  
		movd        xmm0,eax  
		pxor        xmm2,xmm2  
		mov         eax,4B000000h  
		movd        xmm1,eax  
		subps       xmm2,xmm0  
		addss       xmm0,xmm1  
		movd        eax,xmm0  
		subss       xmm1,xmm0  
		cmpeqss     xmm1,xmm2  
		movmskps    edx,xmm1  
		test        edx,1  
		jne         .L14
	.L16:  
		xor         eax,eax  
		jmp         .L15
	.L14:  
		and         eax,1  
		add         eax,1  
		shl         eax,1Eh  
		jmp         .L15
	.L13:  
		cmp         edx,7F800000h  
		jae         .L16  
		cmp         edx,4B000000h  
		je          .L14  
		mov         eax,40000000h
	.L15:  
		mov         dword [rsp+84h],eax  
		mov         edx,dword [rsp+84h]  
		test        edx,edx  
		je          .L17  
		mov         ecx,dword [rdi+4]  
		mov         eax,ecx  
		and         eax,7FFFFFFFh  
		cmp         eax,40862000h  
		jb          .L18  
		and         ecx,80000000h  
		mov         eax,80000000h  
		shr         ecx,1Ch  
		shr         eax,cl  
		add         eax,0FF800000h  
		jmp         .L19
	.L18:  
		mov         qword [rsp+60h],rdi  
		mov         qword [rsp+68h],rdi  
		mov         rax,qword [rsp+68h]  
		movsd       xmm3,qword [rax]  
		unpcklpd    xmm3,xmm3  
		movaps      xmm5,xmm3  
		movaps      oword [rsp+40h],xmm3  
		movaps      oword [rsp+50h],xmm3  
		movapd      xmm0,oword [rel _s_data+110h]  
		movaps      xmm7,xmm3  
		movapd      xmm4,xmm0  
		movdqa      xmm2,oword [rel _s_data+130h]  
		addpd       xmm0,xmm3  
		movapd      xmm1,xmm4  
		subpd       xmm4,xmm0  
		shufps      xmm7,xmm5,0DDh  
		addpd       xmm3,xmm4  
		pand        xmm7,oword [rel _s_data+90h]  
		movapd      xmm6,xmm1  
		pcmpgtd     xmm7,oword [rel _s_data+120h]  
		paddd       xmm0,xmm2  
		pmovmskb    eax,xmm7  
		addpd       xmm1,xmm5  
		movapd      xmm7,oword [rel _s_data+140h]  
		subpd       xmm6,xmm1  
		movapd      xmm4,oword [rel _s_data+150h]  
		addpd       xmm5,xmm6  
		addpd       xmm7,xmm3  
		paddd       xmm1,xmm2  
		movapd      xmm6,xmm5  
		mulpd       xmm7,xmm3  
		addpd       xmm5,xmm4  
		addpd       xmm4,xmm3  
		addpd       xmm7,oword [rel _s_data+160h]  
		movapd      xmm2,xmm6  
		mulpd       xmm5,xmm6  
		mulpd       xmm4,xmm3  
		addpd       xmm2,oword [rel _s_data+140h]  
		psllq       xmm0,34h  
		addpd       xmm5,oword [rel _s_data+170h]  
		addpd       xmm4,oword [rel _s_data+170h]  
		mulpd       xmm2,xmm6  
		mulpd       xmm7,xmm4  
		addpd       xmm2,oword [rel _s_data+160h]  
		movapd      xmm4,xmm3  
		mulpd       xmm5,xmm2  
		addpd       xmm4,oword [rel _s_data+180h]  
		psllq       xmm1,34h  
		movapd      xmm2,oword [rel _s_data+180h]  
		mulpd       xmm4,xmm3  
		addpd       xmm2,xmm6  
		addpd       xmm4,oword [rel _s_data+190h]  
		mulpd       xmm2,xmm6  
		por         xmm0,oword [rel _s_data+1A0h]  
		mulpd       xmm7,xmm4  
		addpd       xmm2,oword [rel _s_data+190h]  
		movdqa      xmm6,oword [rel _s_data+1A0h]  
		mulpd       xmm0,xmm7  
		por         xmm6,xmm1  
		mulpd       xmm5,xmm2  
		cvtpd2ps    xmm0,xmm0  
		mulpd       xmm5,xmm6  
		cvtpd2ps    xmm5,xmm5  
		test        eax,eax  
		shufps      xmm0,xmm5,44h  
		jne         .L20  
		jmp         .L21
	.L20:  
		movaps      xmm3,oword [rsp+40h]  
		movaps      xmm5,oword [rsp+50h]  
		movdqa      xmm7,oword [rel _s_data+1B0h]  
		movdqa      xmm1,oword [rel _s_data+120h]  
		movaps      xmm2,xmm3  
		shufps      xmm3,xmm5,0DDh  
		pxor        xmm6,xmm6  
		pcmpeqd     xmm7,xmm3  
		pcmpgtd     xmm6,xmm3  
		pand        xmm3,oword [rel _s_data+90h]  
		shufps      xmm2,xmm5,88h  
		pcmpgtd     xmm3,xmm1  
		movdqa      xmm4,xmm7  
		pxor        xmm7,xmm3  
		pandn       xmm3,xmm0  
		pandn       xmm6,xmm7  
		movdqa      xmm0,oword [rel _s_data]  
		pand        xmm2,xmm4  
		pand        xmm0,xmm6  
		por         xmm2,xmm3  
		por         xmm0,xmm2
	.L21:  
		mov         rax,qword [rsp+60h]  
		movss       dword [rax],xmm0  
		mov         eax,dword [rdi]
	.L19:  
		and         edx,80000000h  
		or          eax,edx  
		mov         dword [rdi],eax  
		jmp         .L22
	.L17:  
		cmp         r15d,7F800000h  
		jb          .L23  
		jmp         .L22
	.L12:  
		mov         dword [rsp+78h],ebx  
		mov         edx,dword [rsp+78h]  
		pxor        xmm0,xmm0  
		mov         eax,edx  
		and         edx,7F800000h  
		and         eax,7FFFFFFFh  
		cmp         edx,4A800000h  
		ja          .L24  
		movd        xmm0,eax  
		pxor        xmm2,xmm2  
		mov         eax,4B000000h  
		movd        xmm1,eax  
		subps       xmm2,xmm0  
		addss       xmm0,xmm1  
		movd        eax,xmm0  
		subss       xmm1,xmm0  
		cmpeqss     xmm1,xmm2  
		movmskps    edx,xmm1  
		test        edx,1  
		jne         .L25
	.L27:  
		xor         eax,eax  
		jmp         .L26
	.L25:  
		and         eax,1  
		add         eax,1  
		shl         eax,1Eh  
		jmp         .L26
	.L24:  
		cmp         edx,7F800000h  
		jae         .L27  
		cmp         edx,4B000000h  
		je          .L25  
		mov         eax,40000000h
	.L26:  
		mov         dword [rsp+7Ch],eax  
		mov         ecx,ebx  
		mov         eax,80000000h  
		and         ecx,80000000h  
		shr         ecx,1Ch  
		mov         edx,dword [rsp+7Ch]  
		shr         eax,cl  
		and         edx,80000000h  
		and         edx,r13d  
		add         eax,0FF800000h  
		or          edx,eax  
		mov         dword [rdi],edx  
		jmp         .L22
	.L11:  
		mov         r8d,ebx  
		mov         dword [rsp+70h],ebx  
		and         r8d,80000000h  
		mov         edx,dword [rsp+70h]  
		pxor        xmm0,xmm0  
		mov         eax,edx  
		and         edx,7F800000h  
		and         eax,7FFFFFFFh  
		cmp         edx,4A800000h  
		ja          .L28  
		movd        xmm0,eax  
		pxor        xmm2,xmm2  
		mov         eax,4B000000h  
		movd        xmm1,eax  
		subps       xmm2,xmm0  
		addss       xmm0,xmm1  
		movd        eax,xmm0  
		subss       xmm1,xmm0  
		cmpeqss     xmm1,xmm2  
		movmskps    edx,xmm1  
		test        edx,1  
		jne         .L29
	.L31: 
		xor         eax,eax  
		jmp         .L30
	.L29:  
		and         eax,1  
		add         eax,1  
		shl         eax,1Eh  
		jmp         .L30
	.L28:  
		cmp         edx,7F800000h  
		jae         .L31  
		cmp         edx,4B000000h  
		je          .L29  
		mov         eax,40000000h
	.L30:  
		mov         dword [rsp+74h],eax  
		mov         ecx,r8d  
		mov         eax,r13d  
		shr         ecx,1Ch  
		mov         r9d,800000h  
		and         eax,80000000h  
		xor         edx,edx  
		test        dword [rsp+74h],80000000h  
		cmovne      edx,eax  
		shl         r9d,cl  
		add         r9d,0FF800000h  
		or          r9d,edx  
		mov         dword [rdi],r9d  
		test        r8d,r8d  
		je          .L22  
		call        _raise_zerodivide_
	.L22:  
		mov         dword [rdi+4],0FFFFFFFFh
	.L10:  
		cmp         r13d,3F800000h  
		jne         .L32 
		mov         eax,3F800000h  
		mov         dword [rdi+4],0FFFFFFFFh  
		mov         dword [rdi],eax  
		jmp         .L9
	.L32:  
		cmp         r15d,7F800000h  
		jne         .L9  
		mov         dword [rdi+4],0FFFFFFFFh  
		cmp         r14d,3F800000h  
		jne         .L33  
		mov         eax,3F800000h  
		mov         dword [rdi],eax  
		jmp         .L9
	.L33:  
		cmp         r14d,7F800000h  
		jae         .L9  
		cmp         r14d,3F800000h  
		mov         r14d,0  
		seta        r14b  
		mov         eax,7F800000h  
		xor         edx,edx  
		shr         ebx,1Fh  
		cmp         ebx,r14d  
		cmovne      edx,eax  
		mov         dword [rdi],edx
	.L9:  
		inc         rsi  
		add         rdi,8  
		sar         r12d,4  
		cmp         rsi,4  
		jl          .L34  
		movapd      xmm3,oword [rsp+90h]  
		movapd      xmm5,oword [rsp+0A0h]  
		jmp         .L35
	.L1:  
		pxor        xmm0,xmm0
	.L5:  
		movups      xmm6,oword [rsp+0C0h]  
		movups      xmm7,oword [rsp+0B0h]  
		add         rsp,0F0h  
		pop         r15  
		pop         r14  
		pop         r13  
		pop         r12  
		pop         rdi  
		pop         rsi  
		pop         rbx  
		ret
	.L23:
		mov         eax,7FFFFFFFh  
		mov         dword [rdi],eax  
		call        _raise_invalid_  
		nop  
		jmp         .L22  
		nop         word [rax+rax]
	_raise_zerodivide_:  
		pcmpeqd     xmm0,xmm0  
		pxor        xmm1,xmm1  
		pslld       xmm0,1Ch  
		divss       xmm0,xmm1  
		ret  
		nop         word [rax+rax]
	_raise_invalid_:  
		mov         eax,0FF8FFFFFh  
		movd        xmm0,eax  
		addss       xmm0,xmm0  
		ret 