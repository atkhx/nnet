package methods

import (
	"math"
)

func Adagard(learning, eps float64) *adagard {
	return &adagard{learning: learning, eps: eps}
}

type adagard struct {
	learning float64
	eps      float64

	gsum []float64
}

func (t *adagard) Init(weightsCount int) {
	t.gsum = make([]float64, weightsCount)
}

func (t *adagard) GetDelta(k int, gradient float64) float64 {
	t.gsum[k] = t.gsum[k] + gradient*gradient
	return -t.learning / math.Sqrt(t.gsum[k]+t.eps) * gradient
}
