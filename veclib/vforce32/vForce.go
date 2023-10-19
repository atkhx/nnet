package vforce32

/*
#cgo CFLAGS: -I/Library/Developer/CommandLineTools/SDKs/MacOSX13.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers/ -I/Library/Developer/CommandLineTools/SDKs/MacOSX13.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/  -DACCELERATE_NEW_LAPACK=1
#cgo LDFLAGS: -L/usr/lib -framework Accelerate

void vvexpD(float *result, const float *a, const int *n);
void vvrsqrtD(float *result, const float *a, const int *n);
void vvsqrtD(float *result, const float *a, const int *n);

*/
import "C"

func VvexpD(src, dst []float32, length int) {
	C.vvexpD(
		(*C.float)(&dst[0]),
		(*C.float)(&src[0]),
		(*C.int)(pointer(int32(length))),
	)
}

// Vvrsqrt - 1 / √x
func VvrsqrtD(src, dst []float32, length int) {
	C.vvrsqrtD(
		(*C.float)(&dst[0]),
		(*C.float)(&src[0]),
		(*C.int)(pointer(int32(length))),
	)
}

// vvsqrtD - x^2
func VvsqrtD(src, dst []float32, length int) {
	C.vvsqrtD(
		(*C.float)(&dst[0]),
		(*C.float)(&src[0]),
		(*C.int)(pointer(int32(length))),
	)
}

func pointer[T any](v T) *T {
	return &v
}
