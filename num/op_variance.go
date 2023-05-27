package num

func (aData *Data) VarianceByRows(meanData Float64s) *Data {
	oDims := aData.Dims
	oDims.W = 1

	chunkSize := aData.Dims.W
	k := 1.0 / float64(chunkSize-1)

	output := New(oDims, aData)
	output.calcData = func() {
		if meanData == nil {
			mean := aData.MeanByRows()
			mean.Forward()
			meanData = mean.Data
		}

		for i := 0; i < len(output.Data); i++ {
			V := 0.0
			M := meanData[i]
			for _, v := range aData.Data[i*chunkSize : (i+1)*chunkSize] {
				V += (v - M) * (v - M)
			}
			output.Data[i] = k * V
		}
	}

	output.calcGrad = func() {
		for i, G := range output.Grad {
			M := meanData[i]
			for j, v := range aData.Data[i*chunkSize : (i+1)*chunkSize] {
				aData.Grad[i*chunkSize+j] += G * 2.0 * (v - M) * k
			}
		}
	}

	return output
}
