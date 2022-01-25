package lrelu

// Leaky rectified linear unit

func New() *activation {
	return &activation{}
}

type activation struct{}

func (a *activation) Forward(v float64) float64 {
	if v < 0 {
		return 0.01 * v
	}
	return v
}

func (a *activation) Backward(v float64) float64 {
	if v <= 0 {
		return 0.01
	}
	return 1
}
