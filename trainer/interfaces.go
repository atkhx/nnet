package trainer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/loss"
)

const (
	Ro  = 0.95
	Eps = 0.000001
)

type Trainer interface {
	Forward(inputs *data.Data, getLoss loss.GetLossFunc) (*data.Data, loss.LossObject)
	ForwardFn(forwardFn func())
	UpdateWeights()
}

type Net interface {
	Forward(inputs *data.Data) (output *data.Data)
	Backward(lossGradient *data.Data) (gradient *data.Data)
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
