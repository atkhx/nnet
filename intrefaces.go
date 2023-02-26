package nnet

import "github.com/atkhx/nnet/data"

type Layer interface {
	Forward(inputs *data.Matrix) (output *data.Matrix)
}

type WithOutput interface {
	GetOutput() *data.Matrix
}

type WithWeights interface {
	GetWeights() *data.Matrix
}

type WithBiases interface {
	HasBiases() bool
	GetBiases() *data.Matrix
}

type WithGradients interface {
	GetInputGradients() *data.Matrix
}

type WithWeightGradients interface {
	GetWeightGradients() *data.Matrix
}
