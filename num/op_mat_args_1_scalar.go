package num

func (aData *Data) AddScalar(k float64) *Data {
	output := aData.Copy()
	output.calcData = func() {
		output.Data.CopyFrom(aData.Data)
		output.Data.AddScalar(k)
	}
	output.calcGrad = func() {
		for i, g := range output.Grad {
			aData.Grad[i] += g
		}
	}
	return output
}

func (aData *Data) MulScalar(k float64) *Data {
	output := aData.Copy()
	output.calcData = func() {
		output.Data.CopyFrom(aData.Data)
		output.Data.MulScalar(k)
	}
	output.calcGrad = func() {
		for i, g := range output.Grad {
			aData.Grad[i] += g * k
		}
	}
	return output
}
