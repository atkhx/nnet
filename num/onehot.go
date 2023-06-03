package num

func NewSeparateOneHotVectors(featuresCount int) []*Data {
	result := make([]*Data, 0, featuresCount)

	for i := 0; i < featuresCount; i++ {
		data := NewFloat64s(featuresCount)
		data[i] = 1.0
		result = append(result, NewWithValues(NewDims(featuresCount), data))
	}

	return result
}

func NewOneHotVectors(featuresCount int, hots ...int) *Data {
	rows := len(hots)
	data := make(Float64s, 0, featuresCount*rows)

	for row := 0; row < rows; row++ {
		vector := NewFloat64s(featuresCount)
		vector[hots[row]] = 1

		data = append(data, vector...)
	}

	return NewWithValues(NewDims(featuresCount, rows), data)
}
