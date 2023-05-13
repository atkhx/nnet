package num

func (input *Data) MeanByRows() *Data {
	oDims := input.Dims
	oDims.W = 1

	output := New(oDims, input)

	chunkSize := input.Dims.W
	chunksCount := len(input.Data) / chunkSize

	k := 1.0 / float64(chunkSize)

	output.calcData = func() {
		for i := 0; i < chunksCount; i++ {
			output.Data[i] = input.Data[i*chunkSize:(i+1)*chunkSize].Sum() * k
		}
	}

	output.calcGrad = func() {
		input.Grad.AddWeighted(output.Grad, k)
	}

	return output
}
