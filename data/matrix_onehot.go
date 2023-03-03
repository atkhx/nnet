package data

func NewOneHotVectorsMatrix(colsCount int, hots ...int) (outMatrix *Matrix) {
	rowsCount := len(hots)

	data := make([]float64, 0, colsCount*len(hots))

	for row := 0; row < rowsCount; row++ {
		vector := make([]float64, colsCount)
		vector[hots[row]] = 1

		data = append(data, vector...)
	}

	return NewMatrix(colsCount, rowsCount, 1, data)
}

func NewSeparateOneHotVectors(colsCount int) (vectors []*Matrix) {
	for i := 0; i < colsCount; i++ {
		data := make([]float64, colsCount)
		data[i] = 1.0
		vectors = append(vectors, NewMatrix(colsCount, 1, 1, data))
	}
	return
}
