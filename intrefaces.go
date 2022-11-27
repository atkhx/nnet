package nnet

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

	Forward(inputs *data.Data) (output *data.Data)
	Backward(oGrads *data.Data) (iGrads *data.Data)
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
