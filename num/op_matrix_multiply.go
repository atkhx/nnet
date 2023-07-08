package num

import (
	"runtime"
	"sync"
)

func (aData *Data) MatrixMultiply(factor *Data) *Data {
	if aData.Dims.W != factor.Dims.H {
		panic("aData width must be equal factor height")
	}

	izStep := 1
	fzStep := 1

	if aData.Dims.D != factor.Dims.D {
		switch {
		case aData.Dims.D == 1:
			izStep = 0
		case factor.Dims.D == 1:
			fzStep = 0
		default:
			panic("aData's and factor's dept must be equal or one of them must be 1")
		}
	}

	oH := aData.Dims.H
	oW := factor.Dims.W
	oD := aData.Dims.D

	if factor.Dims.D > oD {
		oD = factor.Dims.D
	}

	output := &Data{
		Data:     make(Float64s, oW*oH*oD),
		Grad:     make(Float64s, oW*oH*oD),
		Dims:     Dims{W: oW, H: oH, D: oD},
		srcNodes: Nodes{aData, factor},
	}

	iW, iH := aData.Dims.W, aData.Dims.H
	fW, fH := factor.Dims.W, factor.Dims.H

	iWH := iW * iH
	fWH := fW * fH

	wg := sync.WaitGroup{}
	cn := make(chan struct{}, runtime.GOMAXPROCS(0)*4)

	output.calcData = func() {
		var ozOffset, izOffset, fzOffset int
		wg.Add(oD)
		defer wg.Wait()

		output.Data.Zero()
		for z := 0; z < oD; z++ {
			cn <- struct{}{}
			go func(aData, bData, oData Float64s) {
				mAmB(iW, aData, bData, oData)
				<-cn
				wg.Done()
			}(
				aData.Data[izOffset:izOffset+iWH],
				factor.Data[fzOffset:fzOffset+fWH],
				output.Data[ozOffset:ozOffset+(oW*oH)],
			)

			ozOffset += oW * oH
			izOffset += izStep * iWH
			fzOffset += fzStep * fWH
		}
	}

	output.calcGrad = func() {
		var ozOffset, izOffset, fzOffset int

		wg.Add(oD)
		defer wg.Wait()

		for z := 0; z < oD; z++ {
			cn <- struct{}{}
			go func(iData, iGrad, fData, fGrad, oGrad Float64s) {
				mAmBT(oW, oGrad, fData, iGrad)
				mATmB(iH, iData, oGrad, fGrad)
				<-cn
				wg.Done()
			}(
				aData.Data[izOffset:izOffset+iWH],
				aData.Grad[izOffset:izOffset+iWH],

				factor.Data[fzOffset:fzOffset+fWH],
				factor.Grad[fzOffset:fzOffset+fWH],

				output.Grad[ozOffset:ozOffset+oW*oH],
			)

			ozOffset += oW * oH
			izOffset += izStep * iWH
			fzOffset += fzStep * fWH
		}
	}

	return output
}
