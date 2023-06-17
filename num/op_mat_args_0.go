package num

import "math"

func (aData *Data) Exp() (outMatrix *Data) {
	output := aData.Copy()
	output.calcData = func() {
		for i, x := range aData.Data {
			output.Data[i] = math.Exp(x)
		}
	}
	output.calcGrad = func() {
		for i, y := range output.Data {
			aData.Grad[i] += output.Grad[i] * y
		}
	}
	return output
}

func (aData *Data) Log() (outMatrix *Data) {
	output := aData.Copy()
	output.calcData = func() {
		for i, x := range aData.Data {
			output.Data[i] = math.Log(x)
		}
	}
	output.calcGrad = func() {
		for i, x := range aData.Data {
			aData.Grad[i] += output.Grad[i] / x
		}
	}
	return output
}

func (aData *Data) Pow(n float64) *Data {
	output := aData.Copy()
	output.calcData = func() {
		for i, x := range aData.Data {
			output.Data[i] = math.Pow(x, n)
		}
	}
	output.calcGrad = func() {
		for i, x := range aData.Data {
			aData.Grad[i] += output.Grad[i] * n * math.Pow(x, n-1)
		}
	}
	return output
}

func (aData *Data) Sqrt() (outMatrix *Data) {
	output := aData.Copy()
	output.calcData = func() {
		for i, x := range aData.Data {
			output.Data[i] = math.Sqrt(x)
		}
	}
	output.calcGrad = func() {
		for i, y := range output.Data {
			aData.Grad[i] += output.Grad[i] * 0.5 / y
		}
	}
	return output
}

func (aData *Data) Sigmoid() *Data {
	output := aData.Copy()
	output.calcData = func() {
		for i, x := range aData.Data {
			output.Data[i] = 1.0 / (1.0 + math.Exp(-x))
		}
	}
	output.calcGrad = func() {
		for i, y := range output.Data {
			aData.Grad[i] += output.Grad[i] * y * (1 - y)
		}
	}
	return output
}

func (aData *Data) Tanh() *Data {
	output := aData.Copy()
	output.calcData = func() {
		for i, x := range aData.Data {
			output.Data[i] = math.Tanh(x)
		}
	}
	output.calcGrad = func() {
		for i, y := range output.Data {
			aData.Grad[i] += output.Grad[i] * (1 - y*y)
		}
	}
	return output
}

func (aData *Data) Relu() *Data {
	output := aData.Copy()
	output.calcData = func() {
		for i, x := range aData.Data {
			if x > 0 {
				output.Data[i] = x
			} else {
				output.Data[i] = 0
			}
		}
	}
	output.calcGrad = func() {
		for i, y := range output.Data {
			if y > 0 {
				aData.Grad[i] += output.Grad[i]
			}
		}
	}
	return output
}
