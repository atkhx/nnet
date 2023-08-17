package blas

func MatrixMultiplyAB(aW int, a, b, c []float64, alpha, beta float64) {
	aH := len(a) / aW
	bW := len(b) / aW

	Dgemm(CBLASOrderRowMajor, CBLASNoTrans, CBLASNoTrans,
		aW, aH, bW,
		alpha,

		a, aW,
		b, bW,

		beta,
		c, bW,
	)
}

func MatrixMultiplyAonTransposedB(aW int, a, b, c []float64, alpha, beta float64) {
	aH := len(a) / aW
	bW := len(b) / aW

	Dgemm(CBLASOrderRowMajor, CBLASNoTrans, CBLASTrans,
		aW, aH, bW,
		alpha,

		a, aW,
		b, aW, // change bW to aW

		beta,
		c, bW,
	)
}

func MatrixMultiplyATB(aW int, a, b, c []float64, alpha, beta float64) {
	// aW - ширина уже транспонированной матрицы

	aH := len(a) / aW
	bW := len(b) / aW

	Dgemm(CBLASOrderRowMajor, CBLASTrans, CBLASNoTrans,
		aW, aH, bW,
		alpha,

		a, aH, // change aW to aH
		b, bW,

		beta,
		c, bW,
	)
}
