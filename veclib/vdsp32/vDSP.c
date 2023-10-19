#include <Accelerate/Accelerate.h>

void vclrD(float *vector, int stride, int length) {
    vDSP_vclr(vector, stride, length);
}

void vfillD(const float *vector, float *v, int stride, int length) {
    vDSP_vfill(vector, v, stride, length);
}

void vaddD(float *a, float *b, float *result, int n) {
    vDSP_vadd(a, 1, b, 1, result, 1, n);
}

void vsaddD(float *a, int ia, float *scalar, float *result, int ic, int n) {
    vDSP_vsadd(a, ia, scalar, result, ic, n);
}

void vsmulD(float *a, int ia, float *scalar, float *result, int ic, int n) {
    vDSP_vsmul(a, ia, scalar, result, ic, n);
}

void vmulD(float *a, float *b, float *result, int n) {
    vDSP_vmul(a, 1, b, 1, result, 1, n);
}

void sveD(float *vector, int stride, float *result, int vectorLength) {
    vDSP_sve(vector, stride, result, vectorLength);
}

// https://developer.apple.com/documentation/accelerate/1449854-vdsp_maxvd
void maxvD(float *a, int ia, float *c, int length) {
    vDSP_maxv(a, ia, c, length);
}

// https://developer.apple.com/documentation/accelerate/1450422-vdsp_mtransd
void mtransD(float *a, int ia, float *c, int ic, int m, int n) {
    vDSP_mtrans(a, ia, c, ic, m, n);
}
