package trainer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/data"
)

const (
	Ro  = 0.95
	Eps = 0.000001
)

type Trainer interface {
	Forward(inputs, target *data.Data) *data.Data
	ForwardFn(forwardFn func())
	UpdateWeights()
	GetLossFunc() LossFunc
	GetLossValue() float64
}

type Net interface {
	Forward(inputs *data.Data) (output *data.Data)
	Backward(deltas *data.Data) (gradient *data.Data)
	GetLayersCount() int
	GetLayer(index int) nnet.Layer
}

type LossFunc interface {
	GetError(target, result []float64) (res float64)
	GetDeltas(target, output *data.Data) (deltas *data.Data)
}

type TrainableLayer interface {
	GetWeightsWithGradient() (w, g *data.Data)
	GetBiasesWithGradient() (w, g *data.Data)
	ResetGradients()
	IsTrainable() bool
}

type Method interface {
	Init(weightsCount int)
	GetDelta(k int, gradient float64) float64
}
