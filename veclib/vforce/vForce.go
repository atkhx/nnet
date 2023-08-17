package vforce

/*
#cgo CFLAGS: -I/Library/Developer/CommandLineTools/SDKs/MacOSX13.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers/ -I/Library/Developer/CommandLineTools/SDKs/MacOSX13.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/  -DACCELERATE_NEW_LAPACK=1
#cgo LDFLAGS: -L/usr/lib -framework Accelerate

void vvexpD(double *result, const double *a, const int *n);
void vvrsqrtD(double *result, const double *a, const int *n);
void vvsqrtD(double *result, const double *a, const int *n);

*/
import "C"

func VvexpD(src, dst []float64, length int) {
	C.vvexpD(
		(*C.double)(&dst[0]),
		(*C.double)(&src[0]),
		(*C.int)(pointer(int32(length))),
	)
}

// Vvrsqrt - 1 / √x
func VvrsqrtD(src, dst []float64, length int) {
	C.vvrsqrtD(
		(*C.double)(&dst[0]),
		(*C.double)(&src[0]),
		(*C.int)(pointer(int32(length))),
	)
}

// vvsqrtD - x^2
func VvsqrtD(src, dst []float64, length int) {
	C.vvsqrtD(
		(*C.double)(&dst[0]),
		(*C.double)(&src[0]),
		(*C.int)(pointer(int32(length))),
	)
}

func pointer[T any](v T) *T {
	return &v
}
