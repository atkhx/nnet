package num

func (aData *Data) TriangleLower(zeroVal float64) *Data {
	WH := aData.Dims.W * aData.Dims.W

	output := aData.Copy()
	output.calcData = func() {
		output.Data.Fill(zeroVal)
		for z := 0; z < output.Dims.D; z++ {
			for y := 0; y < output.Dims.H; y++ {
				for x := 0; x <= y; x++ {
					coordinate := z*WH + y*aData.Dims.W + x
					output.Data[coordinate] = aData.Data[coordinate]
				}
			}
		}
	}

	output.calcGrad = func() {
		for z := 0; z < output.Dims.D; z++ {
			for y := 0; y < output.Dims.H; y++ {
				for x := 0; x <= y; x++ {
					coordinate := z*WH + y*aData.Dims.W + x
					aData.Grad[coordinate] += output.Grad[coordinate]
				}
			}
		}
	}
	return output
}
