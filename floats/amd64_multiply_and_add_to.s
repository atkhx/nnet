//go:build amd64 && !noasm

#include "textflag.h"

TEXT ·MultiplyAndAddTo(SB), $24-56
        MOVL    len+8(FP),  CX  // len(dst)
        CMPQ    CX, $0
        JNE     not_empty
        RET
not_empty:
        MOVQ    dst+0(FP),  R11 // dst
        MOVQ    src+24(FP), R12 // src
        MOVUPD  k+48(FP), X4     // k

        MOVQ    CX, DX
        SHRQ    $2, DX // DX = CX/4
        ANDQ    $3, CX // CX = len(dst) % 4

        CMPQ    DX, $0
        JEQ     do_tail

        UNPCKLPD X4, X4
do_double:
        MOVUPD    (R12), X0
        MOVUPD  16(R12), X2

        MOVUPD    (R11), X6 // new
        MOVUPD  16(R11), X8 // new

        MULPD   X4, X0
        MULPD   X4, X2

        ADDPD   X6, X0
        ADDPD   X8, X2

        MOVUPD  X0,   (R11)
        MOVUPD  X2, 16(R11)

        ADDQ    $32, R12
        ADDQ    $32, R11

        DECQ    DX
        JNZ     do_double

        CMPQ    CX, $0
        JNE     do_tail
        RET
do_tail:
        MOVSD   (R12), X0
        MULSD   X4, X0
        ADDSD   (R11), X0
        MOVSD   X0, (R11)

        ADDQ    $8, R12
        ADDQ    $8, R11

        DECQ    CX
        JNZ do_tail
        RET
