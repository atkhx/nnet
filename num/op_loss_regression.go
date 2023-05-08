package num

import "math"

func (input *Data) Regression(targets *Data) *Data {
	input.Dims.MustBeEqual(targets.Dims)
	oDims := input.Dims
	oDims.W = 1

	output := New(oDims, input)
	output.calcData = func() {
		for z := 0; z < input.Dims.D; z++ {
			for y := 0; y < input.Dims.H; y++ {

				r := 0.0
				for x := 0; x < input.Dims.W; x++ {
					c := z*input.Dims.H*input.Dims.W + y*input.Dims.W + x
					r += math.Pow(input.Data[c]-targets.Data[c], 2)
				}
				r *= 0.5

				output.Data[z*input.Dims.H*input.Dims.W+y*input.Dims.W] = r
			}
		}
	}

	output.calcGrad = func() {
		for i, t := range targets.Data {
			input.Grad[i] += input.Data[i] - t
		}
	}

	return output
}
