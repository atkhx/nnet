package num

import (
	"math"
	"math/rand"
)

func (aData *Data) Exp() (outMatrix *Data) {
	output := aData.NewLinkedCopy()
	output.calcData = func() {
		output.Data.Exp()
	}
	output.calcGrad = func() {
		for i, y := range output.Data {
			aData.Grad[i] += output.Grad[i] * y
		}
	}
	return output
}

func (aData *Data) Log() (outMatrix *Data) {
	output := aData.NewLinkedCopy()
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
	output := aData.NewLinkedCopy()
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
	output := aData.NewLinkedCopy()
	output.calcData = func() {
		aData.Data.SqrtTo(output.Data)
		//for i, x := range aData.Data {
		//	output.Data[i] = math.Sqrt(x)
		//}
	}
	output.calcGrad = func() {
		for i, y := range output.Data {
			aData.Grad[i] += output.Grad[i] * 0.5 / y
		}
	}
	return output
}

func (aData *Data) Sigmoid() *Data {
	output := aData.NewLinkedCopy()
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
	output := aData.NewLinkedCopy()
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
	output := aData.NewLinkedCopy()
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

func (aData *Data) Dropout(prob float64) *Data {
	output := &Data{
		Data:          aData.Data,
		Grad:          aData.Grad,
		Dims:          aData.Dims,
		srcNodes:      Nodes{aData},
		calcData:      nil,
		calcGrad:      nil,
		skipResetGrad: true,
	}

	mask10 := aData.Data.CopyZero()
	maskbt := make([]byte, len(mask10))
	maxval := byte(255. * prob)

	output.calcData = func() {
		if _, err := rand.Read(maskbt); err != nil {
			panic("rand.Read: %w" + err.Error())
		}

		mask10.Ones()
		for i, b := range maskbt {
			if b < maxval {
				mask10[i] = 0
			}
		}
		aData.Data.Mul(mask10)
		//aData.Data.MulTo(output.Data, mask10)
	}
	output.calcGrad = func() {
		aData.Grad.Mul(mask10)
		//output.Grad.MulTo(aData.Grad, mask10)
	}
	return output
}
