package num

import (
	"math"
)

func (input *Data) Div(bData *Data) *Data {
	config := BroadCast(input, bData)
	output := New(config.oDims, input, bData)
	square := bData.Data.CopyZero()
	output.calcData = func() {
		config.BroadCast(func(ax, bx, offset int) {
			output.Data[offset] = input.Data[ax] / bData.Data[bx]
		})
	}
	output.calcGrad = func() {
		for k, v := range bData.Data {
			square[k] = -math.Pow(v, -2.0)
		}
		config.BroadCast(func(ax, bx, offset int) {
			iV := input.Data[ax]
			bV := bData.Data[bx]
			gV := output.Grad[offset]

			input.Grad[ax] += gV * (1.0 / bV)
			bData.Grad[bx] += gV * iV * square[bx]
		})
	}
	return output
}
