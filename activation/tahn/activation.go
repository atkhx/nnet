package tahn

import (
	"math"
)

func New() *activation {
	return &activation{}
}

type activation struct{}

func (a *activation) Forward(v float64) float64 {
	e2v := math.Exp(2 * v)
	return (e2v - 1) / (e2v + 1)
}

func (a *activation) Backward(v float64) float64 {
	return 1 - v*v
}
