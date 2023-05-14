package num

func (input *Data) MulScalar(k float64) *Data {
	output := input.Copy()
	output.SetOperation("mulScalar")
	output.calcData = func() {
		output.Data.CopyFrom(input.Data)
		output.Data.MulScalar(k)
	}
	output.calcGrad = func() {
		for i, g := range output.Grad {
			input.Grad[i] += g * k
		}
	}
	return output
}
