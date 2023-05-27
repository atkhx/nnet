package trainer

const (
	Ro  = 0.95
	Eps = 0.000001
)

type Method interface {
	Init(weightsCount int)
	GetDelta(k int, gradient float64) float64
}
