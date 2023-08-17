package num

//
//type mmConfig struct {
//	alpha         float64
//	oData         Float64s
//	oGrad         Float64s
//	skipResetGrad bool
//}
//
//type mmOption func(*mmConfig)
//
//func WithMatrixMultiplyAlpha(alpha float64) mmOption {
//	return func(config *mmConfig) {
//		config.alpha = alpha
//	}
//}
//
//func WithODataAndGrads(oData, oGrad Float64s) mmOption {
//	return func(config *mmConfig) {
//		config.oData = oData
//		config.oGrad = oGrad
//	}
//}
//
//func WithSkipResetGrad(skip bool) mmOption {
//	return func(config *mmConfig) {
//		config.skipResetGrad = skip
//	}
//}
//
//func (aData *Data) splitByDepth() []*Data {
//	var result []*Data
//	iW, iH := aData.Dims.W, aData.Dims.H
//	iWH := iW * iH
//	for z := 0; z < aData.Dims.D; z++ {
//		result = append(result, &Data{
//			Data:          aData.Data[z*iWH : (z+1)*iWH],
//			Grad:          aData.Grad[z*iWH : (z+1)*iWH],
//			Dims:          NewDims(iW, iH),
//			srcNodes:      Nodes{aData},
//			calcData:      func() {},
//			calcGrad:      func() {},
//			skipResetGrad: true,
//		})
//	}
//	return result
//}
//
//func (aData *Data) MatrixMultiply(factor *Data, options ...mmOption) *Data {
//	if aData.Dims.W != factor.Dims.H {
//		panic("aData width must be equal factor height")
//	}
//
//	if factor.Dims.D == 1 && aData.Dims.D == 1 {
//		return aData.MatrixMultiply2D(factor, options...)
//	}
//
//	mW, mH, mD := factor.Dims.W, aData.Dims.H, max(aData.Dims.D, factor.Dims.D)
//	mWH := mW * mH
//
//	//outData := NewFloat64s(mWH * mD)
//	//outGrad := NewFloat64s(mWH * mD)
//
//	var batchesDst []*Data
//
//	if factor.Dims.D != 1 && aData.Dims.D != 1 {
//		if factor.Dims.D != aData.Dims.D {
//			panic("depth of matrices must be equal")
//		}
//
//		matricesA := aData.splitByDepth()
//		matricesB := factor.splitByDepth()
//
//		for z, matrixA := range matricesA {
//			output := matrixA.MatrixMultiply2D(matricesB[z], append([]mmOption{
//				//WithODataAndGrads(
//				//	outData[z*mWH:(z+1)*mWH],
//				//	outGrad[z*mWH:(z+1)*mWH],
//				//),
//				WithSkipResetGrad(false)}, options...)...)
//
//			batchesDst = append(batchesDst, output)
//		}
//	}
//
//	if factor.Dims.D != 1 && aData.Dims.D == 1 {
//		for _, factorD := range factor.splitByDepth() {
//			output := aData.MatrixMultiply2D(factorD, append([]mmOption{
//				//WithODataAndGrads(
//				//	outData[z*mWH:(z+1)*mWH],
//				//	outGrad[z*mWH:(z+1)*mWH],
//				//),
//				WithSkipResetGrad(false)}, options...)...)
//
//			batchesDst = append(batchesDst, output)
//		}
//	}
//
//	if factor.Dims.D == 1 && aData.Dims.D != 1 {
//		for _, matrixA := range aData.splitByDepth() {
//			output := matrixA.MatrixMultiply2D(factor, append([]mmOption{
//				//WithODataAndGrads(
//				//	outData[z*mWH:(z+1)*mWH],
//				//	outGrad[z*mWH:(z+1)*mWH],
//				//),
//				WithSkipResetGrad(false)}, options...)...)
//
//			batchesDst = append(batchesDst, output)
//		}
//	}
//
//	oData := NewFloat64s(mWH * mD)
//	oGrad := NewFloat64s(mWH * mD)
//	return &Data{
//		Data:          oData,
//		Grad:          oGrad,
//		Dims:          NewDims(mW, mH, mD),
//		srcNodes:      batchesDst,
//		skipResetGrad: false,
//		calcData: func() {
//			for z := 0; z < mD; z++ {
//				copy(oData[z*mWH:(z+1)*mWH], batchesDst[z].Data)
//			}
//		},
//		calcGrad: func() {
//			for z := 0; z < mD; z++ {
//				copy(batchesDst[z].Grad, oGrad[z*mWH:(z+1)*mWH])
//			}
//		},
//	}
//}
//
//func (aData *Data) MatrixMultiply2D(factor *Data, options ...mmOption) *Data {
//	if aData.Dims.W != factor.Dims.H {
//		panic("aData width must be equal factor height")
//	}
//
//	if aData.Dims.D != 1 || factor.Dims.D != 1 {
//		panic("depth must be equal 1")
//	}
//
//	oH, oW, oD := aData.Dims.H, factor.Dims.W, aData.Dims.D
//
//	cfg := &mmConfig{
//		alpha: 1.0,
//		oData: NewFloat64s(oW * oH * oD),
//		oGrad: NewFloat64s(oW * oH * oD),
//	}
//
//	for _, option := range options {
//		option(cfg)
//	}
//
//	output := &Data{
//		Data:          cfg.oData,
//		Grad:          cfg.oGrad,
//		Dims:          Dims{W: oW, H: oH, D: oD},
//		srcNodes:      Nodes{aData, factor},
//		skipResetGrad: cfg.skipResetGrad,
//	}
//
//	output.calcData = func() {
//		blas.MatrixMultiplyAB(aData.Dims.W, aData.Data, factor.Data, output.Data, cfg.alpha, 0.0)
//	}
//
//	output.calcGrad = func() {
//		blas.MatrixMultiplyTAB(oW, output.Grad, factor.Data, aData.Grad, cfg.alpha, 1)
//		blas.MatrixMultiplyATB(aData.Dims.H, aData.Data, output.Grad, factor.Grad, cfg.alpha, 1)
//	}
//	return output
//}
