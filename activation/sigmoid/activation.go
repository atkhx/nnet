package sigmoid

import (
	"math"
)

func New() *activation {
	return &activation{}
}

type activation struct{}

func (a *activation) Forward(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (a *activation) Backward(x float64) float64 {
	return x * (1 - x)
}
