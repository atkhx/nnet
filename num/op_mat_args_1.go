package num

import "math"

func (aData *Data) Add(bData *Data) *Data {
	config := BroadCast(aData, bData)
	output := New(config.oDims, aData, bData)
	output.calcData = func() {
		config.BroadCast(func(ax, bx, offset int) {
			output.Data[offset] = aData.Data[ax] + bData.Data[bx]
		})
	}
	output.calcGrad = func() {
		config.BroadCast(func(ax, bx, offset int) {
			aData.Grad[ax] += output.Grad[offset]
			bData.Grad[bx] += output.Grad[offset]
		})
	}
	return output
}

func (aData *Data) Sub(bData *Data) *Data {
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

func (aData *Data) Mul(bData *Data) *Data {
	config := BroadCast(aData, bData)
	output := New(config.oDims, aData, bData)
	output.calcData = func() {
		config.BroadCast(func(ax, bx, offset int) {
			output.Data[offset] = aData.Data[ax] * bData.Data[bx]
		})
	}
	output.calcGrad = func() {
		config.BroadCast(func(ax, bx, offset int) {
			aData.Grad[ax] += output.Grad[offset] * bData.Data[bx]
			bData.Grad[bx] += output.Grad[offset] * aData.Data[ax]
		})
	}
	return output
}

func (aData *Data) Div(bData *Data) *Data {
	config := BroadCast(aData, bData)
	output := New(config.oDims, aData, bData)
	square := bData.Data.CopyZero()
	output.calcData = func() {
		config.BroadCast(func(ax, bx, offset int) {
			output.Data[offset] = aData.Data[ax] / bData.Data[bx]
		})
	}
	output.calcGrad = func() {
		for k, v := range bData.Data {
			square[k] = -math.Pow(v, -2.0)
		}
		config.BroadCast(func(ax, bx, offset int) {
			iV := aData.Data[ax]
			bV := bData.Data[bx]
			gV := output.Grad[offset]

			aData.Grad[ax] += gV * (1.0 / bV)
			bData.Grad[bx] += gV * iV * square[bx]
		})
	}
	return output
}
