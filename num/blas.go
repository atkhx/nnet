package num

// #cgo CFLAGS: -I/Library/Developer/CommandLineTools/SDKs/MacOSX13.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib
// #cgo LDFLAGS: -lcblas
// #include <Accelerate/Accelerate.h>
import "C"

// cblas_dcopy Copies a vector to another vector (double-precision).
func cblas_dcopy(length int, src Float64s, strideSrc int, dst Float64s, strideDst int) {
	// https://developer.apple.com/documentation/accelerate/1513214-cblas_dcopy
	C.cblas_dcopy(
		(C.int)(int32(length)),    // N - Number of elements in the vectors.
		(*C.double)(&src[0]),      // X - Source vector X.
		(C.int)(int32(strideSrc)), // incX - Stride within X. For example, if incX is 7, every 7th element is used.
		(*C.double)(&dst[0]),      // Y - Destination vector Y.
		(C.int)(int32(strideDst)), // incY - Stride within Y. For example, if incY is 7, every 7th element is used.
	)
}

// cblas_daxpy Computes a constant times a vector plus a vector (double-precision).
func cblas_daxpy(length int, alpha float64, src Float64s, strideSrc int, dst Float64s, strideDst int) {
	// https://developer.apple.com/documentation/accelerate/1513298-cblas_daxpy
	C.cblas_daxpy(
		(C.int)(int32(length)),    // N - Number of elements in the vectors.
		(C.double)(alpha),         // alpha - Scaling factor for the values in X.
		(*C.double)(&src[0]),      // X - Input vector X.
		(C.int)(int32(strideSrc)), // incX - Stride within X. For example, if incX is 7, every 7th element is used
		(*C.double)(&dst[0]),      // Y - Input vector Y.
		(C.int)(int32(strideDst)), // incY - Stride within Y. For example, if incY is 7, every 7th element is used.
	)
}

// cblas_dscal Multiplies each element of a vector by a constant (double-precision).
func cblas_dscal(length int, alpha float64, src Float64s, strideSrc int) {
	// https://developer.apple.com/documentation/accelerate/1513084-cblas_dscal
	C.cblas_dscal(
		(C.int)(int32(length)),    // N - Number of elements in the vectors.
		(C.double)(alpha),         // alpha - The constant scaling factor to multiply by.
		(*C.double)(&src[0]),      // X - Input vector X.
		(C.int)(int32(strideSrc)), // incX - Stride within X. For example, if incX is 7, every 7th element is used
	)
}

// cblas_dgemm Multiplies two matrices (double-precision).
func cblas_dgemm(
	order, aTrans, bTrans uint32,
	aW, aH, bW int,
	alpha float64,
	a Float64s, aDim int,
	b Float64s, bDim int,
	beta float64,
	c Float64s, cDim int,
) {
	// https://developer.apple.com/documentation/accelerate/1513282-cblas_dgemm
	C.cblas_dgemm(
		order, // Specifies row-major (C) or column-major (Fortran) data ordering.

		aTrans, // Specifies whether to transpose matrix A.
		bTrans, // Specifies whether to transpose matrix B.

		(C.int)(int32(aH)), // Number of rows in matrices A and C.
		(C.int)(int32(bW)), // Number of columns in matrices B and C.
		(C.int)(int32(aW)), // Number of columns in matrix A; number of rows in matrix B

		(C.double)(alpha),    // alpha - Scaling factor for the product of matrices A and B.
		(*C.double)(&a[0]),   // Matrix A.
		(C.int)(int32(aDim)), // The size of the first dimension of matrix A; if you are passing a matrix A[m][n], the value should be m.

		(*C.double)(&b[0]),   // Matrix B.
		(C.int)(int32(bDim)), // The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.

		(C.double)(beta),     // beta - Scaling factor for matrix C.
		(*C.double)(&c[0]),   // Matrix C.
		(C.int)(int32(cDim)), // The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
	)
}

// cblas_dtrmm Scales a triangular matrix and multiplies it by a matrix.
func cblas_dtrmm(
	order, side, uplo, aTrans, diag uint32,
	bH, bW int,
	alpha float64,

	a Float64s, aDim int,
	b Float64s, bDim int,
) {
	// https://developer.apple.com/documentation/accelerate/1513132-cblas_dtrmm
	C.cblas_dtrmm(
		order,  // Specifies row-major (C) or column-major (Fortran) data ordering.
		side,   // Determines the order in which the matrices should be multiplied.
		uplo,   // Specifies whether to use the upper or lower triangle from the matrix. Valid values are 'U' or 'L'.
		aTrans, // Specifies whether to use matrix A ('N' or 'n') or the transpose of A ('T', 't', 'C', or 'c').
		diag,   // Specifies whether the matrix is unit triangular. Possible values are 'U' (unit triangular) or 'N' (not unit triangular).

		(C.int)(int32(bH)), // Number of rows in matrix B.
		(C.int)(int32(bW)), // Number of columns in matrix B.

		(C.double)(alpha),    // alpha - Scaling factor for the product of matrices A and B.
		(*C.double)(&a[0]),   // Matrix A.
		(C.int)(int32(aDim)), // Leading dimension of matrix A.

		(*C.double)(&b[0]),   // Matrix B. Overwritten by results on return.
		(C.int)(int32(bDim)), // Leading dimension of matrix B.
	)
}

// cblas_dsymm Multiplies a matrix by a symmetric matrix (double-precision).
func cblas_dsymm(
	order, side, uplo uint32,
	aH, bW int,
	alpha float64,

	a Float64s, aDim int,
	b Float64s, bDim int,
	beta float64,
	c Float64s, cDim int,
) {
	// todo check work of this function
	// https://developer.apple.com/documentation/accelerate/1513311-cblas_dsymm
	C.cblas_dsymm(
		order, // Specifies row-major (C) or column-major (Fortran) data ordering.
		side,  // Determines the order in which the matrices should be multiplied.
		uplo,  // Specifies whether to use the upper or lower triangle from the matrix. Valid values are 'U' or 'L'.

		(C.int)(int32(aH)), // Number of rows in matrices A and C.
		(C.int)(int32(bW)), // Number of columns in matrices B and C.

		(C.double)(alpha),    // alpha - Scaling factor for the product of matrices A and B.
		(*C.double)(&a[0]),   // Matrix A.
		(C.int)(int32(aDim)), // The size of the first dimension of matrix A; if you are passing a matrix A[m][n], the value should be m.

		(*C.double)(&b[0]),   // Matrix B.
		(C.int)(int32(bDim)), // The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.

		(C.double)(beta),     // beta - Scaling factor for matrix C.
		(*C.double)(&c[0]),   // Matrix C.
		(C.int)(int32(cDim)), // The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
	)
}

// cblas_dgemv Multiplies a matrix by a vector (double precision).
// This function multiplies A * X (after transposing A, if needed) and multiplies the resulting matrix by alpha.
// It then multiplies vector Y by beta. It stores the sum of these two products in vector Y.
// Thus, it calculates either
// Y←αAX + βY
// with optional use of the transposed form of A.
func cblas_dgemv() {
	// https://developer.apple.com/documentation/accelerate/1513338-cblas_dgemv
	// todo
	panic("implement me")
}

// cblas_dger Multiplies vector X by the transpose of vector Y, then adds matrix A (double precison).
// Computes alpha*x*y' + A.
func cblas_dger() {
	// https://developer.apple.com/documentation/accelerate/1513076-cblas_dger
	// todo
	panic("implement me")
}
