package methods

import (
	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/layer"
)

type Net interface {
	Activate(inputs *data.Data) (output *data.Data)
	Backprop(deltas *data.Data) (gradient *data.Data)
	GetLayersCount() int
	GetLayer(index int) layer.Layer
}

type TrainableLayer interface {
	GetWeightsWithGradient() (w, g *data.Data)
	GetBiasesWithGradient() (w, g *data.Data)
	ResetGradients()
	IsTrainable() bool
}
