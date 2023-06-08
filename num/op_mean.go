package num

func (aData *Data) Mean() *Data {
	k := 1.0 / float64(len(aData.Data))
	output := New(NewDims(), aData)
	output.Name = "mean"
	output.calcData = func() {
		output.Data[0] = aData.Data.Mean()
	}
	output.calcGrad = func() {
		g := output.Grad[0] * k
		for i := range aData.Data {
			aData.Grad[i] += g
		}
	}
	return output
}

func (aData *Data) MeanByRows() *Data {
	oDims := aData.Dims
	oDims.W = 1

	chunkSize := aData.Dims.W
	chunksCount := len(aData.Data) / chunkSize

	k := 1.0 / float64(chunkSize)
	output := New(oDims, aData)
	output.Name = "meanByRows"
	output.calcData = func() {
		for i := 0; i < chunksCount; i++ {
			output.Data[i] = aData.Data[i*chunkSize : (i+1)*chunkSize].Mean()
		}
	}
	output.calcGrad = func() {
		for i := 0; i < chunksCount; i++ {
			aData.Grad[i*chunkSize : (i+1)*chunkSize].AddScalar(output.Grad[i] * k)
		}
	}
	return output
}
