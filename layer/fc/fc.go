package fc

import (
	"math"

	"github.com/atkhx/nnet/data"
)

func New(options ...Option) *FC {
	layer := &FC{}
	layer.gain = data.LinearGain
	layer.batchSize = 1

	applyOptions(layer, defaults...)
	applyOptions(layer, options...)

	layer.Weights = data.NewRandom(
		layer.layerSize,
		layer.inputSize,
		1,
	)

	if layer.gain > 0 {
		fanIn := layer.inputSize * layer.batchSize
		layer.Weights.Data.MulScalar(layer.gain / math.Pow(float64(fanIn), 0.5))
	}

	if layer.WithBiases {
		layer.Biases = data.NewData(layer.layerSize, 1, 1)
	}

	return layer
}

type FC struct {
	layerSize int
	inputSize int
	batchSize int

	inputs, output *data.Data

	// public for easy persist by marshaling network
	Weights *data.Data
	Biases  *data.Data

	gain float64

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
