#include <Accelerate/Accelerate.h>

void vvexpD(double *result, const double *a, const int *n) {
    vvexp(result, a, n);
}

void vvrsqrtD(double *result, const double *a, const int *n) {
    vvrsqrt(result, a, n);
}

void vvsqrtD(double *result, const double *a, const int *n) {
    vvsqrt(result, a, n);
}