package methods

import (
	"math"
)

func Adadelta(ro, eps float64) *adadelta {
	return &adadelta{ro: ro, eps: eps}
}

type adadelta struct {
	ro  float64
	eps float64

	gsum []float64
	xsum []float64
}

func (t *adadelta) Init(weightsCount int) {
	t.gsum = make([]float64, weightsCount)
	t.xsum = make([]float64, weightsCount)
}

func (t *adadelta) GetDelta(k int, gradient float64) float64 {
	t.gsum[k] = t.ro*t.gsum[k] + (1-t.ro)*gradient*gradient

	value := -math.Sqrt((t.xsum[k]+t.eps)/(t.gsum[k]+t.eps)) * gradient

	t.xsum[k] = t.ro*t.xsum[k] + (1-t.ro)*value*value

	return value
}
