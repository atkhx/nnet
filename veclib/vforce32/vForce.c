#include <Accelerate/Accelerate.h>

void vvexpD(float *result, const float *a, const int *n) {
    vvexpf(result, a, n);
}

void vvrsqrtD(float *result, const float *a, const int *n) {
    vvrsqrtf(result, a, n);
}

void vvsqrtD(float *result, const float *a, const int *n) {
    vvsqrtf(result, a, n);
}