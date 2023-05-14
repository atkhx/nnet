package num

func (input *Data) TriangleLower(zeroVal float64) *Data {
	WH := input.Dims.W * input.Dims.W

	output := input.Copy()
	output.SetOperation("triangleLower")
	output.calcData = func() {
		output.Data.Fill(zeroVal)
		for z := 0; z < output.Dims.D; z++ {
			for y := 0; y < output.Dims.H; y++ {
				for x := 0; x <= y; x++ {
					coordinate := z*WH + y*input.Dims.W + x
					output.Data[coordinate] = input.Data[coordinate]
				}
			}
		}
	}

	output.calcGrad = func() {
		for z := 0; z < output.Dims.D; z++ {
			for y := 0; y < output.Dims.H; y++ {
				for x := 0; x <= y; x++ {
					coordinate := z*WH + y*input.Dims.W + x
					input.Grad[coordinate] += output.Grad[coordinate]
				}
			}
		}
	}
	return output
}
