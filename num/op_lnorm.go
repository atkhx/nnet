package num

func (input *Data) LNorm(gamma, beta *Data) *Data {
	mean := input.MeanByRows()
	xSub := input.Sub(mean)

	vars := input.VarianceByRows(mean.Data)
	sqrt := vars.Sqrt()
	xDiv := xSub.Div(sqrt)
	//xMul := xDiv.Mul(gamma)
	xMul := gamma.Mul(xDiv)

	output := xMul.Add(beta)
	outputCalcDataForward := output.calcData

	output.calcData = func() {
		mean.Forward()
		xSub.Forward()

		vars.Forward()
		sqrt.Forward()
		//sqrt.Data.AddScalar(1e-5)

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
