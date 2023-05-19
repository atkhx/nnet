package num

func (input *Data) AddScalar(k float64) *Data {
	output := input.Copy()
	output.calcData = func() {
		output.Data.CopyFrom(input.Data)
		output.Data.AddScalar(k)
	}
	output.calcGrad = func() {
		for i, g := range output.Grad {
			input.Grad[i] += g
		}
	}
	return output
}
