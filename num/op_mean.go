package num

func (input *Data) Mean() *Data {
	output := New(NewDims(), input)

	k := 1.0 / float64(len(input.Data))

	output.calcData = func() {
		output.Data[0] = input.Data.Mean()
	}

	output.calcGrad = func() {
		for i := range input.Data {
			input.Grad[i] += output.Grad[0] * k
		}
	}

	return output
}
