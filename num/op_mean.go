package num

func (input *Data) Mean() *Data {
	output := New(NewDims(), input)

	k := 1.0 / float64(len(input.Data))
	output.calcData = func() {
		r := 0.0
		for _, v := range input.Data {
			r += v
		}
		output.Data[0] = r * k
	}

	output.calcGrad = func() {
		for i, g := range output.Grad {
			input.Grad[i] += g * k
		}
	}

	return output
}

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
