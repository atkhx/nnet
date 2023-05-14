package num

func (input *Data) VarianceByRows(meanData Float64s) *Data {
	oDims := input.Dims
	oDims.W = 1

	chunkSize := input.Dims.W
	k := 1.0 / float64(chunkSize-1)

	output := New(oDims, input)
	output.calcData = func() {
		if meanData == nil {
			meanData = input.MeanByRows().Data
		}

		for i := 0; i < len(output.Data); i++ {
			V := 0.0
			M := meanData[i]
			for _, v := range input.Data[i*chunkSize : (i+1)*chunkSize] {
				V += (v - M) * (v - M)
			}
			output.Data[i] = V
		}
	}

	output.calcGrad = func() {
		for i, G := range output.Grad {
			M := meanData[i]
			for j, v := range input.Data[i*chunkSize : (i+1)*chunkSize] {
				input.Grad[i*chunkSize+j] += G * 2.0 * (v - M) * k
			}
		}
	}

	return output
}
