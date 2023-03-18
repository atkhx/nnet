package fc

import (
	"math"

	"github.com/atkhx/nnet/data"
)

func New(options ...Option) *FC {
	layer := &FC{}
	applyOptions(layer, defaults...)
	applyOptions(layer, options...)

	//layer.Weights = data.NewRandomMinMax(
	//	layer.layerSize,
	//	layer.inputSize,
	//	1,
	//	-1,
	//	1,
	//)

	layer.Weights = data.NewRandom(
		layer.layerSize,
		layer.inputSize,
		1,
	)

	//layer.Weights.Data.MulScalar(0.01)
	layer.Weights.Data.MulScalar(data.ReLuGain / math.Pow(float64(layer.inputSize), 0.5))

	//
	//wk := math.Sqrt(2) / math.Sqrt(float64(layer.Weights.Data.Len()))
	//for k := range layer.Weights.Data.Data {
	//	layer.Weights.Data.Data[k] *= wk
	//}

	if layer.WithBiases {
		layer.Biases = data.NewData(layer.layerSize, 1, 1)
	}

	return layer
}

type FC struct {
	layerSize int
	inputSize int

	inputs, output *data.Data

	// public for easy persist by marshaling network
	Weights *data.Data
	Biases  *data.Data

	WithBiases bool
}

func (l *FC) Forward(inputs *data.Data) *data.Data {
	l.inputs = inputs
	l.output = l.inputs.MatrixMultiply(l.Weights)

	if l.WithBiases {
		l.output = l.output.AddRowVector(l.Biases)
	}

	return l.output
}

func (l *FC) GetOutput() *data.Data {
	return l.output
}

func (l *FC) GetWeights() *data.Data {
	return l.Weights
}

func (l *FC) GetBiases() *data.Data {
	return l.Biases
}

func (l *FC) HasBiases() bool {
	return l.WithBiases
}

func (l *FC) GetInputGradients() (g *data.Volume) {
	return l.inputs.Grad
}
