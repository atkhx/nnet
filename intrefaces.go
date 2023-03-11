package nnet

import "github.com/atkhx/nnet/data"

type Layer interface {
	Forward(inputs *data.Data) (output *data.Data)
}

type WithOutput interface {
	GetOutput() *data.Data
}

type WithWeights interface {
	GetWeights() *data.Data
}

type WithBiases interface {
	HasBiases() bool
	GetBiases() *data.Data
}

type WithGradients interface {
	GetInputGradients() *data.Volume
}
