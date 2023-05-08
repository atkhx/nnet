package num

func (input *Data) ConcatRows(bData *Data) *Data {
	if input.Dims.H != bData.Dims.H {
		panic("height dimension must be equals")
	}

	if input.Dims.D != bData.Dims.D {
		panic("depth dimension must be equals")
	}

	oDims := NewDims(
		input.Dims.W+bData.Dims.W,
		input.Dims.H,
		input.Dims.D,
	)

	output := New(oDims, input, bData)

	output.calcData = func() {
		oOffset := 0
		iOffset := 0
		bOffset := 0

		for i := 0; i < output.Dims.D*output.Dims.H; i++ {
			// copy input data
			copy(output.Data[oOffset:oOffset+input.Dims.W], input.Data[iOffset:iOffset+input.Dims.W])
			// copy bData data
			oOffset += input.Dims.W
			copy(output.Data[oOffset:oOffset+bData.Dims.W], bData.Data[bOffset:bOffset+bData.Dims.W])

			iOffset += input.Dims.W
			bOffset += bData.Dims.W
			oOffset += bData.Dims.W
		}
	}

	output.calcGrad = func() {
		oOffset := 0
		iOffset := 0
		bOffset := 0

		for i := 0; i < output.Dims.D*output.Dims.H; i++ {
			// copy input grads
			input.Grad[iOffset : iOffset+input.Dims.W].Add(output.Grad[oOffset : oOffset+input.Dims.W])
			// copy bData grads
			oOffset += input.Dims.W
			bData.Grad[bOffset : bOffset+bData.Dims.W].Add(output.Grad[oOffset : oOffset+bData.Dims.W])

			iOffset += input.Dims.W
			bOffset += bData.Dims.W
			oOffset += bData.Dims.W
		}
	}

	return output
}
