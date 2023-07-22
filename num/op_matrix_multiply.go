package num

type mmConfig struct {
	alpha float64
}

type mmOption func(*mmConfig)

func WithMatrixMultiplyAlpha(alpha float64) mmOption {
	return func(config *mmConfig) {
		config.alpha = alpha
	}
}

func (aData *Data) MatrixMultiply(factor *Data, options ...mmOption) *Data {
	if aData.Dims.W != factor.Dims.H {
		panic("aData width must be equal factor height")
	}

	cfg := &mmConfig{alpha: 1.0}
	for _, option := range options {
		option(cfg)
	}
	alpha := cfg.alpha

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
	oWH := oW * oH

	if factor.Dims.D == 1 {
		output.calcData = func() {
			matrixMultiplyAB(aData.Dims.W, aData.Data, factor.Data, output.Data, alpha, 0.0)
		}

		output.calcGrad = func() {
			matrixMultiplyABTransposed(oW, output.Grad, factor.Data, aData.Grad, alpha, 1)
			matrixMultiplyATransposedB(aData.Dims.H*aData.Dims.D, aData.Data, output.Grad, factor.Grad, alpha, 1)
		}
		return output
	}

	output.calcData = func() {
		var ozOffset, izOffset, fzOffset int
		for z := 0; z < oD; z++ {
			matrixMultiplyAB(iW,
				aData.Data[izOffset:izOffset+iWH],
				factor.Data[fzOffset:fzOffset+fWH],
				output.Data[ozOffset:ozOffset+oWH], alpha, 0)

			ozOffset += oWH
			izOffset += izStep * iWH
			fzOffset += fzStep * fWH
		}
	}

	output.calcGrad = func() {
		var ozOffset, izOffset, fzOffset int
		for z := 0; z < oD; z++ {
			matrixMultiplyABTransposed(oW,
				output.Grad[ozOffset:ozOffset+oWH],
				factor.Data[fzOffset:fzOffset+fWH],
				aData.Grad[izOffset:izOffset+iWH],
				alpha, 1)

			matrixMultiplyATransposedB(iH,
				aData.Data[izOffset:izOffset+iWH],
				output.Grad[ozOffset:ozOffset+oWH],
				factor.Grad[fzOffset:fzOffset+fWH],
				alpha, 1)

			ozOffset += oWH
			izOffset += izStep * iWH
			fzOffset += fzStep * fWH
		}
	}

	return output
}
