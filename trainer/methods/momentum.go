package methods

func Momentum(learning, momentum float64) *momentumSGD {
	return &momentumSGD{learning: learning, momentum: momentum}
}

type momentumSGD struct {
	momentum float64
	learning float64

	gsum []float64
}

func (t *momentumSGD) Init(weightsCount int) {
	t.gsum = make([]float64, weightsCount)
}

func (t *momentumSGD) GetDelta(k int, gradient float64) float64 {
	t.gsum[k] = t.gsum[k]*t.momentum - t.learning*gradient

	return t.gsum[k]
}
