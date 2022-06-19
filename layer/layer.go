package layer

import "github.com/atkhx/nnet/data"

type Layer interface {
	InitDataSizes(
		inputWidth,
		inputHeight,
		inputDepth int,
	) (
		outputWidth,
		outputHeight,
		outputDepth int,
	)

	Activate(inputs *data.Data) (output *data.Data)
	Backprop(deltas *data.Data) (nextDeltas *data.Data)
}

type WithOutput interface {
	GetOutput() *data.Data
}

type WithWeights interface {
	GetWeights() *data.Data
}

type WithBiases interface {
	GetBiases() *data.Data
}

type WithGradients interface {
	GetInputGradients() *data.Data
}

type WithWeightGradients interface {
	GetWeightGradients() *data.Data
}
