package data

import "fmt"

func MatrixColFloats(aColsCount, aRowsCount, colIndex int, a []float64) []float64 {
	res := make([]float64, aRowsCount)
	for rowIndex := 0; rowIndex < aRowsCount; rowIndex++ {
		res[rowIndex] = a[rowIndex*aColsCount+colIndex]
	}

	return res
}

func MatrixRowFloats(aColsCount, rowIndex int, a []float64) []float64 {
	rowOffset := rowIndex * aColsCount
	return a[rowOffset : rowOffset+aColsCount]
}

func MatrixRowFloatsCopy(aColsCount, rowIndex int, a []float64) []float64 {
	src := MatrixRowFloats(aColsCount, rowIndex, a)
	dst := make([]float64, len(src))
	copy(dst, src)
	return dst
}

func MatrixMultiply(
	aColsCount, aRowsCount int, a []float64,
	bColsCount, bRowsCount int, b []float64,
) (
	rColsCount, rRowsCount int, r []float64,
) {
	if aColsCount != bRowsCount {
		panic(fmt.Sprintf("aColsCount != bRowsCount: %d != %d", aColsCount, bRowsCount))
	}

	rColsCount, rRowsCount = bColsCount, aRowsCount

	r = make([]float64, rColsCount*rRowsCount)

	for weightIndex := 0; weightIndex < bColsCount; weightIndex++ {
		bFloats := MatrixColFloats(bColsCount, bRowsCount, weightIndex, b)

		for inputIndex := 0; inputIndex < aRowsCount; inputIndex++ {
			aFloats := MatrixRowFloats(aColsCount, inputIndex, a)

			abDot := 0.0
			for i, aV := range aFloats {
				abDot += aV * bFloats[i]
			}

			r[inputIndex*bColsCount+weightIndex] = abDot
		}
	}

	return
}

func MatrixAddRowVector(aColsCount, aRowsCount int, a, b []float64) (out []float64) {
	if aColsCount != len(b) {
		panic(fmt.Sprintf("invalid vector length: expected %d, actual %d", aColsCount, len(b)))
	}

	out = make([]float64, 0, aColsCount*aRowsCount)

	for rowIndex := 0; rowIndex < aRowsCount; rowIndex++ {
		row := MatrixRowFloatsCopy(aColsCount, rowIndex, a)
		for i, v := range b {
			row[i] += v
		}
		out = append(out, row...)
	}

	return
}

func MatrixTranspose(
	aColsCount, aRowsCount int, a []float64,
) (
	rColsCount, rRowsCount int, r []float64,
) {
	rColsCount, rRowsCount = aRowsCount, aColsCount
	r = make([]float64, rColsCount*rRowsCount)

	for row := 0; row < rRowsCount; row++ {
		for col := 0; col < rColsCount; col++ {
			r[row*rColsCount+col] = a[col*aColsCount+row]
		}
	}
	return
}
