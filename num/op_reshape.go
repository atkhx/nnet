package num

func (input *Data) Reshape(dims Dims) *Data {
	if input.Dims.Size() != dims.Size() {
		panic("total dimension size must be equal with original")
	}

	output := &Data{
		Data:     input.Data,
		Grad:     input.Grad,
		Dims:     dims,
		srcNodes: Nodes{input},
		calcData: nil,
		calcGrad: nil,
	}

	//output := New(dims, input)
	output.calcData = func() {
		//output.Data.CopyFrom(input.Data)
	}
	output.calcGrad = func() {
		//input.Grad.CopyFrom(output.Grad)
	}
	return output
}
