package methods

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/data"
)

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
