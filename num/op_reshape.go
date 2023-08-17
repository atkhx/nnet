package num

func (aData *Data) Reshape(dims Dims) *Data {
	if aData.Dims.Size() != dims.Size() {
		panic("total dimension size must be equal with original")
	}

	output := &Data{
		Data:          aData.Data,
		Grad:          aData.Grad,
		Dims:          dims,
		srcNodes:      Nodes{aData},
		calcData:      nil,
		calcGrad:      nil,
		skipResetGrad: true,
	}
	// output := New(dims, aData)
	output.calcData = func() {
		// output.Data.CopyFrom(aData.Data)
	}
	output.calcGrad = func() {
		// aData.Grad.CopyFrom(output.Grad)
	}
	return output
}
