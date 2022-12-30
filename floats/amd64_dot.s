//go:build amd64 && !noasm

#include "textflag.h"

TEXT ·Dot(SB), $24-56
        MOVQ    $0, CX
        MOVQ    CX, X8

        MOVL    len+8(FP),  CX  // len(sliceA)
        CMPQ    CX, $0
        JEQ     do_sum_and_ret
not_empty:
        MOVQ    sliceA+0(FP),  R11 // sliceA
        MOVQ    sliceB+24(FP), R12 // sliceB

        MOVQ    CX, DX // DX = len(sliceA)
        SHRQ    $2, DX // DX = DX/4
        ANDQ    $3, CX // CX = len(sliceA) % 4

        CMPQ    DX, $0
        JEQ     do_tail
do_double:
        MOVUPD    (R11), X0
        MOVUPD  16(R11), X2

        MOVUPD    (R12), X4
        MOVUPD  16(R12), X6

        MULPD   X4, X0
        MULPD   X6, X2

        ADDPD   X0, X8
        ADDPD   X2, X8

        ADDQ    $32, R12
        ADDQ    $32, R11

        DECQ    DX
        JNZ     do_double

        CMPQ    CX, $0
        JEQ     do_sum_and_ret
do_tail:
        MOVSD   (R11), X0
        MOVSD   (R12), X4

        MULSD   X4, X0
        ADDSD   X0, X8

        ADDQ    $8, R12
        ADDQ    $8, R11

        DECQ    CX
        JNZ do_tail
do_sum_and_ret:
        MOVHPD X8, res+48(FP)
        MOVQ res+48(FP), X10
        ADDPD X8, X10
        MOVQ X10, res+48(FP)
        RET
