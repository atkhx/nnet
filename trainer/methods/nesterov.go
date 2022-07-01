package methods

func Nesterov(learning, momentum float64) *nesterov {
	return &nesterov{learning: learning, momentum: momentum}
}

type nesterov struct {
	momentum float64
	learning float64

	gsum []float64
}

func (t *nesterov) Init(weightsCount int) {
	t.gsum = make([]float64, weightsCount)
}

func (t *nesterov) GetDelta(k int, gradient float64) float64 {
	value := t.gsum[k]

	t.gsum[k] = t.gsum[k]*t.momentum + t.learning*gradient

	return t.momentum*value - (1.0+t.momentum)*t.gsum[k]
}
