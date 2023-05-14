package num

func (input *Data) LNorm(gamma, beta *Data) *Data {
	mean := input.MeanByRows()
	vars := input.VarianceByRows(mean.Data)
	sqrt := vars.Sqrt()
	xSub := input.Sub(mean)

	xDiv := xSub.Div(sqrt)
	xMul := xDiv.Mul(gamma)

	output := xMul.Add(beta)
	outputCalcDataOrigin := output.calcData

	output.calcData = func() {
		mean.Forward()
		vars.Forward()
		sqrt.Forward()
		xSub.Forward()
		xDiv.Forward()
		xMul.Forward()
		outputCalcDataOrigin()
	}

	// todo correct calcGrad func

	return output
}
