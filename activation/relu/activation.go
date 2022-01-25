package relu

func New() *activation {
	return &activation{}
}

type activation struct{}

func (a *activation) Forward(v float64) float64 {
	if v < 0 {
		v = 0
	}
	return v
}

func (a *activation) Backward(v float64) float64 {
	if v <= 0 {
		return 0
	}
	return 1
}
