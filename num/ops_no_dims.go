package num

import "math"

func (input *Data) Exp() (outMatrix *Data) {
	output := input.Copy()
	output.calcData = func() {
		for i, x := range input.Data {
			output.Data[i] = math.Exp(x)
		}
	}
	output.calcGrad = func() {
		for i, y := range output.Data {
			input.Grad[i] += output.Grad[i] * y
		}
	}
	return output
}

func (input *Data) Log() (outMatrix *Data) {
	output := input.Copy()
	output.calcData = func() {
		for i, x := range input.Data {
			output.Data[i] = math.Log(x)
		}
	}
	output.calcGrad = func() {
		for i, x := range input.Data {
			input.Grad[i] += output.Grad[i] / x
		}
	}
	return output
}

func (input *Data) Pow(n float64) *Data {
	output := input.Copy()
	output.calcData = func() {
		for i, x := range input.Data {
			output.Data[i] = math.Pow(x, n)
		}
	}
	output.calcGrad = func() {
		for i, x := range input.Data {
			input.Grad[i] += output.Grad[i] * n * math.Pow(x, n-1)
		}
	}
	return output
}

func (input *Data) Sqrt() (outMatrix *Data) {
	output := input.Copy()
	output.calcData = func() {
		for i, x := range input.Data {
			output.Data[i] = math.Sqrt(x)
		}
	}
	output.calcGrad = func() {
		for i, x := range input.Data {
			input.Grad[i] += output.Grad[i] * 0.5 * math.Pow(x, -0.5)
		}
	}
	return output
}

func (input *Data) Sigmoid() *Data {
	output := input.Copy()
	output.calcData = func() {
		for i, x := range input.Data {
			output.Data[i] = 1.0 / (1.0 + math.Exp(-x))
		}
	}
	output.calcGrad = func() {
		for i, y := range output.Data {
			input.Grad[i] += output.Grad[i] * y * (1 - y)
		}
	}
	return output
}

func (input *Data) Tanh() *Data {
	output := input.Copy()
	output.calcData = func() {
		for i, x := range input.Data {
			output.Data[i] = math.Tanh(x)
		}
	}
	output.calcGrad = func() {
		for i, y := range output.Data {
			input.Grad[i] += output.Grad[i] * (1 - y*y)
		}
	}
	return output
}

func (input *Data) Relu() *Data {
	output := input.Copy()
	output.calcData = func() {
		for i, x := range input.Data {
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
				input.Grad[i] += output.Grad[i]
			}
		}
	}
	return output
}
