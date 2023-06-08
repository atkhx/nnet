package num

func (aData *Data) LNorm(gamma, beta *Data) *Data {
	mean := aData.MeanByRows()
	xSub := aData.Sub(mean)
	vars := aData.VarianceByRows(mean)

	sqrt := vars.Sqrt()
	sqrtEps := sqrt.AddScalar(0.000001)
	xDiv := xSub.Div(sqrtEps)
	xMul := gamma.Mul(xDiv)

	return xMul.Add(beta)
}
