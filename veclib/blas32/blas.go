package blas32

// #cgo CFLAGS: -I/Library/Developer/CommandLineTools/SDKs/MacOSX13.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers/ -I/Library/Developer/CommandLineTools/SDKs/MacOSX13.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/  -DACCELERATE_NEW_LAPACK=1
// #cgo LDFLAGS: -lcblas
// #include <Accelerate/Accelerate.h>
import "C"

// test

// Copy Copies a vector to another vector
func Copy(length int, src []float32, strideSrc int, dst []float32, strideDst int) {
	// https://developer.apple.com/documentation/accelerate/1513168-cblas_scopy
	C.cblas_scopy(
		(C.int)(int32(length)),    // N - Number of elements in the vectors.
		(*C.float)(&src[0]),       // X - Source vector X.
		(C.int)(int32(strideSrc)), // incX - Stride within X. For example, if incX is 7, every 7th element is used.
		(*C.float)(&dst[0]),       // Y - Destination vector Y.
		(C.int)(int32(strideDst)), // incY - Stride within Y. For example, if incY is 7, every 7th element is used.
	)
}

// Axpy Computes a constant times a vector plus a vector
func Axpy(length int, alpha float32, src []float32, strideSrc int, dst []float32, strideDst int) {
	// https://developer.apple.com/documentation/accelerate/1513188-cblas_saxpy
	C.cblas_saxpy(
		(C.int)(int32(length)),    // N - Number of elements in the vectors.
		(C.float)(alpha),          // alpha - Scaling factor for the values in X.
		(*C.float)(&src[0]),       // X - Input vector X.
		(C.int)(int32(strideSrc)), // incX - Stride within X. For example, if incX is 7, every 7th element is used
		(*C.float)(&dst[0]),       // Y - Input vector Y.
		(C.int)(int32(strideDst)), // incY - Stride within Y. For example, if incY is 7, every 7th element is used.
	)
}

// Scal Multiplies each element of a vector by a constant
func Scal(length int, alpha float32, src []float32, strideSrc int) {
	// https://developer.apple.com/documentation/accelerate/1513354-cblas_sscal
	C.cblas_sscal(
		(C.int)(int32(length)),    // N - Number of elements in the vectors.
		(C.float)(alpha),          // alpha - The constant scaling factor to multiply by.
		(*C.float)(&src[0]),       // X - Input vector X.
		(C.int)(int32(strideSrc)), // incX - Stride within X. For example, if incX is 7, every 7th element is used
	)
}

// Gemm Multiplies two matrices
func Gemm(
	order, aTrans, bTrans uint32,
	aW, aH, bW int,
	alpha float32,
	a []float32, aDim int,
	b []float32, bDim int,
	beta float32,
	c []float32, cDim int,
) {
	// https://developer.apple.com/documentation/accelerate/1513264-cblas_sgemm
	C.cblas_sgemm(
		order, // Specifies row-major (C) or column-major (Fortran) data ordering.

		aTrans, // Specifies whether to transpose matrix A.
		bTrans, // Specifies whether to transpose matrix B.

		(C.int)(int32(aH)), // Number of rows in matrices A and C.
		(C.int)(int32(bW)), // Number of columns in matrices B and C.
		(C.int)(int32(aW)), // Number of columns in matrix A; number of rows in matrix B

		(C.float)(alpha),     // alpha - Scaling factor for the product of matrices A and B.
		(*C.float)(&a[0]),    // Matrix A.
		(C.int)(int32(aDim)), // The size of the first dimension of matrix A; if you are passing a matrix A[m][n], the value should be m.

		(*C.float)(&b[0]),    // Matrix B.
		(C.int)(int32(bDim)), // The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.

		(C.float)(beta),      // beta - Scaling factor for matrix C.
		(*C.float)(&c[0]),    // Matrix C.
		(C.int)(int32(cDim)), // The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
	)
}

// Trmm Scales a triangular matrix and multiplies it by a matrix.
func Trmm(
	order, side, uplo, aTrans, diag uint32,
	bH, bW int,
	alpha float32,

	a []float32, aDim int,
	b []float32, bDim int,
) {
	// https://developer.apple.com/documentation/accelerate/1513037-cblas_strmm
	C.cblas_strmm(
		order,  // Specifies row-major (C) or column-major (Fortran) data ordering.
		side,   // Determines the order in which the matrices should be multiplied.
		uplo,   // Specifies whether to use the upper or lower triangle from the matrix. Valid values are 'U' or 'L'.
		aTrans, // Specifies whether to use matrix A ('N' or 'n') or the transpose of A ('T', 't', 'C', or 'c').
		diag,   // Specifies whether the matrix is unit triangular. Possible values are 'U' (unit triangular) or 'N' (not unit triangular).

		(C.int)(int32(bH)), // Number of rows in matrix B.
		(C.int)(int32(bW)), // Number of columns in matrix B.

		(C.float)(alpha),     // alpha - Scaling factor for the product of matrices A and B.
		(*C.float)(&a[0]),    // Matrix A.
		(C.int)(int32(aDim)), // Leading dimension of matrix A.

		(*C.float)(&b[0]),    // Matrix B. Overwritten by results on return.
		(C.int)(int32(bDim)), // Leading dimension of matrix B.
	)
}
