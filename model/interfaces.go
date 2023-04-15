package model

type Layer interface {
	Forward()
	Backward()

	Buffers() (output, oGrads []float64)
}

type Updatable interface {
	ForUpdate() [][2][]float64
}

type WithGrads interface {
	ResetGrads()
}

type LossFn interface {
	GetLoss(target, actual []float64) (loss float64)
	GetGrads(target, actual, oGrads []float64)
}
