package num

import (
	"math"
)

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
			gV := output.Grad[offset]
			if gV == 0 {
				return
			}

			if bV := bData.Data[bx]; bV != 0 {
				aData.Grad[ax] += gV / bV
			}

			if iV := aData.Data[ax]; iV != 0 {
				bData.Grad[bx] += gV * iV * square[bx]
			}
		})
	}
	return output
}
