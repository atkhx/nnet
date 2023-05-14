package num

func (input *Data) ConcatRows(bData ...*Data) *Data {
	if input.Dims.H != bData[0].Dims.H {
		panic("height dimension must be equals")
	}

	if input.Dims.D != bData[0].Dims.D {
		panic("depth dimension must be equals")
	}

	width := 0
	for _, b := range bData {
		width += b.Dims.W
	}

	oDims := NewDims(
		input.Dims.W+width,
		input.Dims.H,
		input.Dims.D,
	)

	output := New(oDims, append(Nodes{input}, bData...)...)

	output.calcData = func() {
		oOffset := 0
		iOffset := 0
		bOffset := 0

		for i := 0; i < output.Dims.D*output.Dims.H; i++ {
			// copy input data
			copy(output.Data[oOffset:oOffset+input.Dims.W], input.Data[iOffset:iOffset+input.Dims.W])
			oOffset += input.Dims.W

			for _, b := range bData {
				// copy bData data
				copy(output.Data[oOffset:oOffset+b.Dims.W], b.Data[bOffset:bOffset+b.Dims.W])
				oOffset += b.Dims.W
			}

			iOffset += input.Dims.W
			bOffset += bData[0].Dims.W
		}
	}

	output.calcGrad = func() {
		oOffset := 0
		iOffset := 0
		bOffset := 0

		for i := 0; i < output.Dims.D*output.Dims.H; i++ {
			// copy input grads
			input.Grad[iOffset : iOffset+input.Dims.W].Add(output.Grad[oOffset : oOffset+input.Dims.W])
			oOffset += input.Dims.W
			for _, b := range bData {
				// copy bData data
				b.Grad[bOffset : bOffset+b.Dims.W].Add(output.Grad[oOffset : oOffset+b.Dims.W])
				oOffset += b.Dims.W
			}

			iOffset += input.Dims.W
			bOffset += bData[0].Dims.W
		}
	}

	return output
}
