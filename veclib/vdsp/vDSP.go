package vdsp

/*
#cgo CFLAGS: -I/Library/Developer/CommandLineTools/SDKs/MacOSX13.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers/ -I/Library/Developer/CommandLineTools/SDKs/MacOSX13.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/  -DACCELERATE_NEW_LAPACK=1
#cgo LDFLAGS: -L/usr/lib -framework Accelerate

void vclrD(double *vector, int stride, int length);
void vfillD(const double *vector, double *v, int stride, int length);
void vaddD(double *a, double *b, double *result, int n);
void vsaddD(double *a, int ia, double *scalar, double *result, int ic, int n);
void vsmulD(double *a, int ia, double *scalar, double *result, int ic, int n);
void vmulD(double *a, double *b, double *result, int n);
void sveD(double *vector, int stride, double *result, int vectorLength);
void maxvD(double *a, int ia, double *c, int length);

void mtransD(double *a, int ia, double *c, int ic, int m, int n);

*/
import "C"

func VclrD(a []float64, stride, length int) {
	C.vclrD(
		(*C.double)(&a[0]),
		(C.int)(int32(stride)),
		(C.int)(int32(length)),
	)
}

func VclrDAll(a []float64) {
	C.vclrD(
		(*C.double)(&a[0]),
		1,
		(C.int)(len(a)),
	)
}

func VfillD(v float64, a []float64, stride, length int) {
	C.vfillD(
		(*C.double)(&v),
		(*C.double)(&a[0]),
		(C.int)(stride),
		(C.int)(length),
	)
}

func VfillDAll(v float64, a []float64) {
	C.vfillD(
		(*C.double)(&v),
		(*C.double)(&a[0]),
		1,
		(C.int)(len(a)),
	)
}

var one = (*C.double)(pointer(1.0))

func VfillDAllOnes(a []float64) {
	C.vfillD(
		one,
		(*C.double)(&a[0]),
		1,
		(C.int)(len(a)),
	)
}

func VaddD(a, b, r []float64, length int) {
	C.vaddD(
		(*C.double)(&a[0]),
		(*C.double)(&b[0]),
		(*C.double)(&r[0]),
		C.int(length),
	)
}

func VsaddD(
	src []float64, srcStride int,
	dst []float64, dstStride int,
	scalar float64, length int,
) {
	C.vsaddD(
		(*C.double)(&src[0]),
		(C.int)(int32(srcStride)),
		(*C.double)(&scalar),
		(*C.double)(&dst[0]),
		(C.int)(int32(dstStride)),
		C.int(length),
	)
}

func VsmulD(
	src []float64, srcStride int,
	dst []float64, dstStride int,
	scalar float64, length int,
) {
	C.vsmulD(
		(*C.double)(&src[0]),
		(C.int)(int32(srcStride)),
		(*C.double)(&scalar),
		(*C.double)(&dst[0]),
		(C.int)(int32(dstStride)),
		C.int(length),
	)
}

func VmulD(a, b, r []float64, length int) {
	C.vmulD(
		(*C.double)(&a[0]),
		(*C.double)(&b[0]),
		(*C.double)(&r[0]),
		C.int(length),
	)
}

func SveD(a []float64, stride, length int) (result float64) {
	C.sveD(
		(*C.double)(&a[0]),
		C.int(stride),
		(*C.double)(&result),
		C.int(length),
	)

	return
}

func MaxvD(a []float64, stride, length int) (result float64) {
	C.maxvD(
		(*C.double)(&a[0]),
		C.int(stride),
		(*C.double)(&result),
		C.int(length),
	)

	return
}

func MtransD(
	a []float64, ia int,
	c []float64, ic int,
	m, n int,
) {
	// a - matrix
	// ia - stride a
	// c - out matrix
	// ic - stride c
	// M - Number of rows in C and the number of columns in A.
	// N - Number of columns in C and the number of rows in A.
	C.mtransD(
		(*C.double)(&a[0]),
		(C.int)(int32(ia)),
		(*C.double)(&c[0]),
		(C.int)(int32(ic)),
		(C.int)(int32(m)),
		(C.int)(int32(n)),
	)
}

func pointer[T any](v T) *T {
	return &v
}
