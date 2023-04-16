package layer

type Layer interface {
	Forward()
	Backward()

	Compile(inputs, iGrads []float64) ([]float64, []float64)
}

type Updatable interface {
	ForUpdate() [][2][]float64
}

type WithGrads interface {
	ResetGrads()
}
