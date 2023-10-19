package vdsp32

/*
#cgo CFLAGS: -I/Library/Developer/CommandLineTools/SDKs/MacOSX13.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers/ -I/Library/Developer/CommandLineTools/SDKs/MacOSX13.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/  -DACCELERATE_NEW_LAPACK=1
#cgo LDFLAGS: -L/usr/lib -framework Accelerate

void vclrD(float *vector, int stride, int length);
void vfillD(const float *vector, float *v, int stride, int length);
void vaddD(float *a, float *b, float *result, int n);
void vsaddD(float *a, int ia, float *scalar, float *result, int ic, int n);
void vsmulD(float *a, int ia, float *scalar, float *result, int ic, int n);
void vmulD(float *a, float *b, float *result, int n);
void sveD(float *vector, int stride, float *result, int vectorLength);
void maxvD(float *a, int ia, float *c, int length);

void mtransD(float *a, int ia, float *c, int ic, int m, int n);

*/
import "C"

func VclrD(a []float32, stride, length int) {
	C.vclrD(
		(*C.float)(&a[0]),
		(C.int)(int32(stride)),
		(C.int)(int32(length)),
	)
}

func VclrDAll(a []float32) {
	C.vclrD(
		(*C.float)(&a[0]),
		1,
		(C.int)(len(a)),
	)
}

func VfillD(v float32, a []float32, stride, length int) {
	C.vfillD(
		(*C.float)(&v),
		(*C.float)(&a[0]),
		(C.int)(stride),
		(C.int)(length),
	)
}

func VfillDAll(v float32, a []float32) {
	C.vfillD(
		(*C.float)(&v),
		(*C.float)(&a[0]),
		1,
		(C.int)(len(a)),
	)
}

var one = (*C.float)(pointer(float32(1.0)))

func VfillDAllOnes(a []float32) {
	C.vfillD(
		one,
		(*C.float)(&a[0]),
		1,
		(C.int)(len(a)),
	)
}

func VaddD(a, b, r []float32, length int) {
	C.vaddD(
		(*C.float)(&a[0]),
		(*C.float)(&b[0]),
		(*C.float)(&r[0]),
		C.int(length),
	)
}

func VsaddD(
	src []float32, srcStride int,
	dst []float32, dstStride int,
	scalar float32, length int,
) {
	C.vsaddD(
		(*C.float)(&src[0]),
		(C.int)(int32(srcStride)),
		(*C.float)(&scalar),
		(*C.float)(&dst[0]),
		(C.int)(int32(dstStride)),
		C.int(length),
	)
}

func VsmulD(
	src []float32, srcStride int,
	dst []float32, dstStride int,
	scalar float32, length int,
) {
	C.vsmulD(
		(*C.float)(&src[0]),
		(C.int)(int32(srcStride)),
		(*C.float)(&scalar),
		(*C.float)(&dst[0]),
		(C.int)(int32(dstStride)),
		C.int(length),
	)
}

func VmulD(a, b, r []float32, length int) {
	C.vmulD(
		(*C.float)(&a[0]),
		(*C.float)(&b[0]),
		(*C.float)(&r[0]),
		C.int(length),
	)
}

func SveD(a []float32, stride, length int) (result float32) {
	C.sveD(
		(*C.float)(&a[0]),
		C.int(stride),
		(*C.float)(&result),
		C.int(length),
	)

	return
}

func MaxvD(a []float32, stride, length int) (result float32) {
	C.maxvD(
		(*C.float)(&a[0]),
		C.int(stride),
		(*C.float)(&result),
		C.int(length),
	)

	return
}

func MtransD(
	a []float32, ia int,
	c []float32, ic int,
	m, n int,
) {
	// a - matrix
	// ia - stride a
	// c - out matrix
	// ic - stride c
	// M - Number of rows in C and the number of columns in A.
	// N - Number of columns in C and the number of rows in A.
	C.mtransD(
		(*C.float)(&a[0]),
		(C.int)(int32(ia)),
		(*C.float)(&c[0]),
		(C.int)(int32(ic)),
		(C.int)(int32(m)),
		(C.int)(int32(n)),
	)
}

func pointer[T any](v T) *T {
	return &v
}
