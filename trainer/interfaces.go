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
	Forward(inputs *data.Matrix, getLoss func(output *data.Matrix) *data.Matrix) (loss *data.Matrix)
	updateWeights()
}

type Net interface {
	Forward(inputs *data.Matrix) (output *data.Matrix)
	GetLayersCount() int
	GetLayer(index int) nnet.Layer
}

type Method interface {
	Init(weightsCount int)
	GetDelta(k int, gradient float64) float64
}
