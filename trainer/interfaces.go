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
	Forward(inputs *data.Data, getLoss func(output *data.Data) *data.Data) (loss *data.Data)
	updateWeights()
}

type Net interface {
	Forward(inputs *data.Data) (output *data.Data)
	GetLayersCount() int
	GetLayer(index int) nnet.Layer
}

type Method interface {
	Init(weightsCount int)
	GetDelta(k int, gradient float64) float64
}
