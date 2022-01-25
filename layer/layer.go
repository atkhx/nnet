package layer

import "github.com/atkhx/nnet/data"

type Layer interface {
	InitDataSizes(w, h, d int) (int, int, int)

	Activate(inputs *data.Data) (output *data.Data)
	Backprop(deltas *data.Data) (nextDeltas *data.Data)
}

type LayerWithOutput interface {
	GetOutput() *data.Data
}

type LayerWithWeights interface {
	GetWeights() *data.Data
}

type LayerWithBiases interface {
	GetBiases() *data.Data
}

type LayerWithGradients interface {
	GetInputGradients() *data.Data
}
