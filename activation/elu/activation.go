package elu

import (
	"math"
)

func New(k float64) *activation {
	return &activation{K: k}
}

type activation struct {
	K float64
}

func (a *activation) Forward(v float64) float64 {
	if v <= 0 {
		return a.K * (math.Exp(v) - 1)
	}
	return v
}

func (a *activation) Backward(v float64) float64 {
	if v <= 0 {
		return a.K * math.Exp(v)
	}
	return 1
}
