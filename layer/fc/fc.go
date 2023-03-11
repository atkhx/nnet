package fc

import (
	"math"

	"github.com/atkhx/nnet/data"
)

func New(options ...Option) *FC {
	layer := &FC{}
	applyOptions(layer, defaults...)
	applyOptions(layer, options...)

	layer.Weights = data.NewRandomMinMax(
		layer.layerSize,
		layer.inputSize,
		1,
		-1,
		1,
	)

	wk := math.Sqrt(2) / math.Sqrt(float64(layer.Weights.Data.Len()))
	for k := range layer.Weights.Data.Data {
		layer.Weights.Data.Data[k] *= wk
	}

	if layer.WithBiases {
		layer.Biases = data.NewRandom(layer.layerSize, 1, 1)
		layer.Biases.Data.Fill(0)
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
