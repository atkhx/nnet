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
