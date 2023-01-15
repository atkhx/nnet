package trainer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/data"
)

const (
	Ro  = 0.95
	Eps = 0.000001

	l1Decay = 0.001
	l2Decay = 0.001
)

type Trainer interface {
	Forward(inputs, target *data.Data) *data.Data
	UpdateWeights()
}

type Net interface {
	Forward(inputs *data.Data) (output *data.Data)
	Backward(deltas *data.Data) (gradient *data.Data)
	GetLayersCount() int
	GetLayer(index int) nnet.Layer
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
