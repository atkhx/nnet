package num

func (aData *Data) Sub(bData *Data) *Data {
	if len(aData.Data) == len(bData.Data) {
		return aData.SubEqualLength(bData)
	}

	if aData.Dims.W == bData.Dims.W && bData.Dims.H == 1 && bData.Dims.D == 1 {
		return aData.SubRowVector(bData)
	}

	config := BroadCast(aData, bData)
	output := New(config.oDims, aData, bData)
	output.calcData = func() {
		config.BroadCast(func(ax, bx, offset int) {
			output.Data[offset] = aData.Data[ax] - bData.Data[bx]
		})
	}
	output.calcGrad = func() {
		config.BroadCast(func(ax, bx, offset int) {
			aData.Grad[ax] += output.Grad[offset]
			bData.Grad[bx] -= output.Grad[offset]
		})
	}

	return output
}

func (aData *Data) SubEqualLength(bData *Data) *Data {
	if len(aData.Data) != len(bData.Data) {
		panic("len(aData.Data) != len(bData.Data)")
	}
	config := BroadCast(aData, bData)
	output := New(config.oDims, aData, bData)
	output.calcData = func() {
		output.Data.CopyFrom(aData.Data)
		output.Data.Sub(bData.Data)
	}
	output.calcGrad = func() {
		aData.Grad.Add(output.Grad)
		bData.Grad.Sub(output.Grad)
	}
	return output
}

func (aData *Data) SubRowVector(bData *Data) *Data {
	isValid := aData.Dims.W == bData.Dims.W && bData.Dims.H == 1 && bData.Dims.D == 1
	if !isValid {
		panic("invalid dimensions")
	}

	config := BroadCast(aData, bData)
	output := New(config.oDims, aData, bData)
	output.calcData = func() {
		output.Data.CopyFrom(aData.Data)
		offset := 0

		for i := 0; i < aData.Dims.D*aData.Dims.H; i++ {
			output.Data[offset : offset+aData.Dims.W].Sub(bData.Data)
			offset += aData.Dims.W
		}
	}

	output.calcGrad = func() {
		aData.Grad.Add(output.Grad)
		offset := 0

		for i := 0; i < aData.Dims.D*aData.Dims.H; i++ {
			bData.Grad.Sub(output.Grad[offset : offset+aData.Dims.W])
			offset += aData.Dims.W
		}
	}

	return output
}
