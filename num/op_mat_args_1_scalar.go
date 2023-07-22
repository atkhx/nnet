package num

func (aData *Data) MulScalar(k float64) *Data {
	output := aData.NewLinkedCopy()
	output.calcData = func() {
		output.Data.CopyFrom(aData.Data)
		output.Data.MulScalar(k)
	}
	output.calcGrad = func() {
		aData.Grad.AddWeighted(output.Grad, k)
	}
	return output
}
