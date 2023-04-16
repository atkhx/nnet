package num

func NewOneHotVectors(colsCount int, hots ...int) Float64s {
	rowsCount := len(hots)

	data := make(Float64s, 0, colsCount*len(hots))

	for row := 0; row < rowsCount; row++ {
		vector := make(Float64s, colsCount)
		vector[hots[row]] = 1.0

		data = append(data, vector...)
	}

	return data
}
