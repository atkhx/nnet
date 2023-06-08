package num

func (aData *Data) Transpose() *Data {
	WH := aData.Dims.W * aData.Dims.H

	output := aData.Copy()
	output.Name = "transpose " + aData.Name
	output.Dims.W = aData.Dims.H
	output.Dims.H = aData.Dims.W
	output.calcData = func() {
		for d := 0; d < len(aData.Data); d += WH {
			for y := 0; y < aData.Dims.H; y++ {
				for x := 0; x < aData.Dims.W; x++ {
					output.Data[d+x*aData.Dims.H+y] = aData.Data[d+y*aData.Dims.W+x]
				}
			}
		}
	}
	output.calcGrad = func() {
		for d := 0; d < len(aData.Data); d += WH {
			for y := 0; y < aData.Dims.H; y++ {
				for x := 0; x < aData.Dims.W; x++ {
					aData.Grad[d+y*aData.Dims.W+x] += output.Grad[d+x*aData.Dims.H+y]
				}
			}
		}
	}
	return output
}
