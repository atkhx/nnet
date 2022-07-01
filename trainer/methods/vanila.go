package methods

func VanilaSGD(learning float64) *vanila {
	return &vanila{learning: learning}
}

type vanila struct {
	learning float64
}

func (t *vanila) Init(weightsCount int) {
}

func (t *vanila) GetDelta(k int, gradient float64) float64 {
	return -t.learning * gradient
}
