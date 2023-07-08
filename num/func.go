package num

// #cgo CFLAGS: -I/Library/Developer/CommandLineTools/SDKs/MacOSX13.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers -I/Users/andrey.tikhonov/go/src/github.com/atkhx/nnet-veclib -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib
// #cgo LDFLAGS: -lcblas
// #include <Accelerate/Accelerate.h>
import (
	"C"
)

func max[T int | int64 | float64](a, b T) T {
	if a > b {
		return a
	}
	return b
}

func mAmB(aW int, a, b, c Float64s) {
	aH := int32(len(a) / aW)
	M := (*C.int)(&aH)
	bH := int32(len(b) / aW)
	N := (*C.int)(&bH)
	k := int32(aW)
	K := (*C.int)(&k)

	C.cblas_dgemm(
		101, 111, 111,
		*M, // 3, // M - Number of rows in matrices A and C.
		*N, // 3, // N - Number of columns in matrices B and C.
		*K, // 3, // K - Number of columns in matrix A; number of rows in matrix B
		1,  // alpha - Scaling factor for the product of matrices A and B.
		(*C.double)(&a[0]),
		*K, // 3, // The size of the first dimension of matrix A; if you are passing a matrix A[m][n], the value should be m.
		(*C.double)(&b[0]),
		*N, // 3, // The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.
		1,  // beta - Scaling factor for matrix C.
		(*C.double)(&c[0]),
		*N, // 3, // The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
	)
}

func pointer[T any](v T) *T {
	return &v
}

func mAmBT(AWidth int, a, b, c Float64s) {
	aW := (*C.int)(pointer(int32(AWidth)))
	aH := (*C.int)(pointer(int32(len(a) / AWidth)))
	bW := (*C.int)(pointer(int32(len(b) / AWidth)))

	K := aW // K - Number of columns in matrix A; number of rows in matrix B
	M := aH // M - Number of rows in matrices A and C.
	N := bW // N - Number of columns in matrices B and C.

	C.cblas_dgemm(
		101, 111, 112,
		*M, // 3, // M - Number of rows in matrices A and C.
		*N, // 3, // N - Number of columns in matrices B and C.
		*K, // 3, // K - Number of columns in matrix A; number of rows in matrix B
		1,  // alpha - Scaling factor for the product of matrices A and B.
		(*C.double)(&a[0]),
		*K, // 3, // The size of the first dimension of matrix A; if you are passing a matrix A[m][n], the value should be m.
		(*C.double)(&b[0]),
		*K, // *N, // 3, // The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.
		1,  // beta - Scaling factor for matrix C.
		(*C.double)(&c[0]),
		*N, // 3, // The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
	)
}

// AWidth - ширина уже транспонированной матрицы
func mATmB(AWidth int, a, b, c Float64s) {
	aW := (*C.int)(pointer(int32(AWidth)))
	aH := (*C.int)(pointer(int32(len(a) / AWidth)))
	bW := (*C.int)(pointer(int32(len(b) / AWidth)))

	K := aW // K - Number of columns in matrix A; number of rows in matrix B
	M := aH // M - Number of rows in matrices A and C.
	N := bW // N - Number of columns in matrices B and C.

	C.cblas_dgemm(
		101, 112, 111,
		*M, // 3, // M - Number of rows in matrices A and C.
		*N, // 3, // N - Number of columns in matrices B and C.
		*K, // 3, // K - Number of columns in matrix A; number of rows in matrix B
		1,  // alpha - Scaling factor for the product of matrices A and B.
		(*C.double)(&a[0]),
		*M, // *K, // 3, // The size of the first dimension of matrix A; if you are passing a matrix A[m][n], the value should be m.
		(*C.double)(&b[0]),
		*N, // 3, // The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.
		1,  // beta - Scaling factor for matrix C.
		(*C.double)(&c[0]),
		*N, // 3, // The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
	)
}

func mm_tr_lower(aW int, a, b, c Float64s) {
	oW := len(b) / aW

	for y := 0; y < aW; y++ {
		for x, v := range a[y*aW : y*aW+y+1] {
			if v != 0 {
				axpyUnitary(v, b[x*oW:x*oW+oW], c[y*oW:y*oW+oW])
			}
		}
	}
}

func axpyUnitary(alpha float64, x, y []float64) {
	for i, v := range x {
		y[i] += alpha * v
	}
}

func transpose(aW, aH int, aData Float64s) Float64s {
	oData := aData.CopyZero()
	transposeTo(aW, aH, aData, oData)
	return oData
}

func transposeTo(aW, aH int, aData, oData Float64s) {
	WH := aW * aH
	for d := 0; d < len(aData); d += WH {
		for y := 0; y < aH; y++ {
			for x := 0; x < aW; x++ {
				oData[d+x*aH+y] = aData[d+y*aW+x]
			}
		}
	}
}

func GetMinMaxValues(data []float64) (min, max float64) {
	for i := 0; i < len(data); i++ {
		if i == 0 || min > data[i] {
			min = data[i]
		}
		if i == 0 || max < data[i] {
			max = data[i]
		}
	}
	return
}

func dot(iData, fData Float64s) (v float64) {
	for i, iV := range iData {
		v += iV * fData[i]
	}
	return v
}
