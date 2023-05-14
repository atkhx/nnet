package num

func (input *Data) Transpose() (outMatrix *Data) {
	WH := input.Dims.W * input.Dims.H

	output := input.Copy()
	output.Dims.W = input.Dims.H
	output.Dims.H = input.Dims.W
	output.calcData = func() {
		for d := 0; d < len(input.Data); d += WH {
			for y := 0; y < input.Dims.H; y++ {
				for x := 0; x < input.Dims.W; x++ {
					output.Data[d+x*input.Dims.H+y] = input.Data[d+y*input.Dims.W+x]
				}
			}
		}
	}
	output.calcGrad = func() {
		for d := 0; d < len(input.Data); d += WH {
			for y := 0; y < input.Dims.H; y++ {
				for x := 0; x < input.Dims.W; x++ {
					input.Grad[d+y*input.Dims.W+x] += output.Grad[d+x*input.Dims.H+y]
				}
			}
		}
	}
	return output
}
