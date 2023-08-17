package num

import (
	"github.com/atkhx/nnet/veclib/vdsp"
)

func (aData *Data) Transpose() *Data {
	IW, IH := aData.Dims.W, aData.Dims.H
	WH := aData.Dims.W * aData.Dims.H

	output := aData.NewLinkedCopy()
	output.Dims.W = aData.Dims.H
	output.Dims.H = aData.Dims.W
	output.calcData = func() {
		for d := 0; d < len(aData.Data); d += WH {
			vdsp.MtransD(aData.Data[d:d+WH], 1, output.Data[d:d+WH], 1, IW, IH)
			//for y := 0; y < aData.Dims.H; y++ {
			//	for x := 0; x < aData.Dims.W; x++ {
			//		output.Data[d+x*aData.Dims.H+y] = aData.Data[d+y*aData.Dims.W+x]
			//	}
			//}
		}
	}

	aGrad := NewFloat64s(WH)
	output.calcGrad = func() {
		for d := 0; d < len(aData.Data); d += WH {
			vdsp.MtransD(output.Grad[d:d+WH], 1, aGrad, 1, IH, IW)
			aData.Grad[d : d+WH].Add(aGrad)
			//for y := 0; y < aData.Dims.H; y++ {
			//	for x := 0; x < aData.Dims.W; x++ {
			//		aData.Grad[d+y*aData.Dims.W+x] += output.Grad[d+x*aData.Dims.H+y]
			//	}
			//}
		}
	}
	return output
}
