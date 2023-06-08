package num

import "math"

func (aData *Data) Regression(targets *Data) *Data {
	aData.Dims.MustBeEqual(targets.Dims)
	oDims := aData.Dims
	oDims.W = 1

	output := New(oDims, aData)
	output.Name = "regression"
	output.calcData = func() {
		for z := 0; z < aData.Dims.D; z++ {
			for y := 0; y < aData.Dims.H; y++ {

				r := 0.0
				for x := 0; x < aData.Dims.W; x++ {
					c := z*aData.Dims.H*aData.Dims.W + y*aData.Dims.W + x
					r += math.Pow(aData.Data[c]-targets.Data[c], 2)
				}
				r *= 0.5

				output.Data[z*aData.Dims.H*aData.Dims.W+y*aData.Dims.W] = r
			}
		}
	}

	output.calcGrad = func() {
		for i, t := range targets.Data {
			aData.Grad[i] += aData.Data[i] - t
		}
	}

	return output
}
