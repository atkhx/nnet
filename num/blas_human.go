package num

func vectorCopy(src, dst Float64s) {
	cblas_dcopy(max(len(src), len(dst)), src, 1, dst, 1)
}

func vectorAddWeighted(src, dst Float64s, alpha float64) {
	cblas_daxpy(max(len(src), len(dst)), alpha, src, 1, dst, 1)
}

func matrixMultiplyAB(aW int, a, b, c Float64s, alpha, beta float64) {
	aH := len(a) / aW
	bW := len(b) / aW

	cblas_dgemm(CBLASOrderRowMajor, CBLASNoTrans, CBLASNoTrans,
		aW, aH, bW,
		alpha,

		a, aW,
		b, bW,

		beta,
		c, bW,
	)
}

func matrixMultiplyABTransposed(aW int, a, b, c Float64s, alpha, beta float64) {
	aH := len(a) / aW
	bW := len(b) / aW

	cblas_dgemm(CBLASOrderRowMajor, CBLASNoTrans, CBLASTrans,
		aW, aH, bW,
		alpha,

		a, aW,
		b, aW, // change bW to aW

		beta,
		c, bW,
	)
}

func matrixMultiplyATransposedB(aW int, a, b, c Float64s, alpha, beta float64) {
	// aW - ширина уже транспонированной матрицы

	aH := len(a) / aW
	bW := len(b) / aW

	cblas_dgemm(CBLASOrderRowMajor, CBLASTrans, CBLASNoTrans,
		aW, aH, bW,
		alpha,

		a, aH, // change aW to aH
		b, bW,

		beta,
		c, bW,
	)
}
