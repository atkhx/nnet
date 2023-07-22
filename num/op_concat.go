package num

func (aData *Data) ConcatRows(bData ...*Data) *Data {
	if aData.Dims.H != bData[0].Dims.H {
		panic("height dimension must be equals")
	}

	if aData.Dims.D != bData[0].Dims.D {
		panic("depth dimension must be equals")
	}

	width := 0
	for _, b := range bData {
		width += b.Dims.W
	}

	oDims := NewDims(
		aData.Dims.W+width,
		aData.Dims.H,
		aData.Dims.D,
	)

	output := New(oDims, append(Nodes{aData}, bData...)...)
	output.calcData = func() {
		oOffset := 0
		iOffset := 0
		bOffset := 0

		for i := 0; i < output.Dims.D*output.Dims.H; i++ {
			// copy aData data
			copy(output.Data[oOffset:oOffset+aData.Dims.W], aData.Data[iOffset:iOffset+aData.Dims.W])
			oOffset += aData.Dims.W

			for _, b := range bData {
				// copy bData data
				copy(output.Data[oOffset:oOffset+b.Dims.W], b.Data[bOffset:bOffset+b.Dims.W])
				oOffset += b.Dims.W
			}

			iOffset += aData.Dims.W
			bOffset += bData[0].Dims.W
		}
	}

	output.calcGrad = func() {
		oOffset := 0
		iOffset := 0
		bOffset := 0

		for i := 0; i < output.Dims.D*output.Dims.H; i++ {
			// copy aData grads
			aData.Grad[iOffset : iOffset+aData.Dims.W].Add(output.Grad[oOffset : oOffset+aData.Dims.W])
			oOffset += aData.Dims.W

			for _, b := range bData {
				// copy bData data
				b.Grad[bOffset : bOffset+b.Dims.W].Add(output.Grad[oOffset : oOffset+b.Dims.W])
				oOffset += b.Dims.W
			}

			iOffset += aData.Dims.W
			bOffset += bData[0].Dims.W
		}
	}

	return output
}
