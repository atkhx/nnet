package num

func (aData *Data) LNorm(gamma, beta *Data) *Data {
	mean := aData.MeanByRows()
	xSub := aData.Sub(mean)
	vars := aData.VarianceByRows(mean.Data)

	sqrt := vars.Sqrt()
	sqrtEps := sqrt.AddScalar(0.000001)
	xDiv := xSub.Div(sqrtEps)
	xMul := gamma.Mul(xDiv)

	output := xMul.Add(beta)
	outputCalcDataForward := output.calcData

	output.calcData = func() {
		mean.Forward()
		xSub.Forward()

		vars.Forward()
		sqrt.Forward()
		sqrtEps.Forward()

		xDiv.Forward()
		xMul.Forward()
		outputCalcDataForward()
	}

	outputCalcDataBackward := output.calcGrad
	output.calcGrad = func() {
		outputCalcDataBackward()

		xMul.Backward()
		xDiv.Backward()

		sqrtEps.Backward()
		sqrt.Backward()
		vars.Backward()
		xSub.Backward()
		mean.Backward()
	}

	return output
}
