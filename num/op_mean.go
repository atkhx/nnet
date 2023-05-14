package num

func (input *Data) Mean() *Data {
	output := New(NewDims(), input)

	k := 1.0 / float64(len(input.Data))
	output.SetOperation("mean")
	output.calcData = func() {
		r := 0.0
		for _, v := range input.Data {
			r += v
		}
		output.Data[0] = r * k
	}

	output.calcGrad = func() {
		for i, ig := range input.Grad {
			input.Grad[i] = ig + output.Grad[0]*k
		}
	}

	return output
}
