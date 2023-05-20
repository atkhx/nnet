package num

func (aData *Data) LNorm(gamma, beta *Data) *Data {
	mean := aData.MeanByRows()
	xSub := aData.Sub(mean)
	vars := aData.VarianceByRows(mean.Data)

	sqrt := vars.Sqrt()
	xDiv := xSub.Div(sqrt)
	xMul := gamma.Mul(xDiv)

	output := xMul.Add(beta)
	outputCalcDataForward := output.calcData

	output.calcData = func() {
		mean.Forward()
		xSub.Forward()

		vars.Forward()
		sqrt.Forward()

		xDiv.Forward()
		xMul.Forward()
		outputCalcDataForward()
	}

	outputCalcDataBackward := output.calcGrad
	output.calcGrad = func() {
		outputCalcDataBackward()

		xMul.Backward()
		xDiv.Backward()

		sqrt.Backward()
		vars.Backward()
		xSub.Backward()
		mean.Backward()
	}

	return output
}
