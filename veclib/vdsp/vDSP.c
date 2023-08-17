#include <Accelerate/Accelerate.h>

void vclrD(double *vector, int stride, int length) {
    vDSP_vclrD(vector, stride, length);
}

void vfillD(const double *vector, double *v, int stride, int length) {
    vDSP_vfillD(vector, v, stride, length);
}

void vaddD(double *a, double *b, double *result, int n) {
    vDSP_vaddD(a, 1, b, 1, result, 1, n);
}

void vsaddD(double *a, int ia, double *scalar, double *result, int ic, int n) {
    vDSP_vsaddD(a, ia, scalar, result, ic, n);
}

void vsmulD(double *a, int ia, double *scalar, double *result, int ic, int n) {
    vDSP_vsmulD(a, ia, scalar, result, ic, n);
}

void vmulD(double *a, double *b, double *result, int n) {
    vDSP_vmulD(a, 1, b, 1, result, 1, n);
}

void sveD(double *vector, int stride, double *result, int vectorLength) {
    vDSP_sveD(vector, stride, result, vectorLength);
}

// https://developer.apple.com/documentation/accelerate/1449854-vdsp_maxvd
void maxvD(double *a, int ia, double *c, int length) {
    vDSP_maxvD(a, ia, c, length);
}

// https://developer.apple.com/documentation/accelerate/1450422-vdsp_mtransd
void mtransD(double *a, int ia, double *c, int ic, int m, int n) {
    vDSP_mtransD(a, ia, c, ic, m, n);
}
