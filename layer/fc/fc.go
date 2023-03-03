package fc

import (
	"github.com/atkhx/nnet/data"
)

func New(options ...Option) *FC {
	layer := &FC{}
	applyOptions(layer, defaults...)
	applyOptions(layer, options...)

	layer.Weights = data.NewMatrixRandom(
		layer.layerSize,
		layer.inputSize,
		1,
	)

	if layer.WithBiases {
		layer.Biases = data.NewMatrixRandom(layer.layerSize, 1, 1)
	}

	return layer
}

type FC struct {
	layerSize int
	inputSize int

	inputs, output *data.Matrix

	// public for easy persist by marshaling network
	Weights *data.Matrix
	Biases  *data.Matrix

	WithBiases bool
}

func (l *FC) Forward(inputs *data.Matrix) *data.Matrix {
	//fmt.Println("-------")
	//fmt.Println("fc layer")
	//fmt.Println("inputs", inputs.GetDims())
	l.inputs = inputs
	l.output = l.inputs.MatrixMultiply(l.Weights)
	//fmt.Println("output", l.output.GetDims())

	if l.WithBiases {
		l.output = l.output.AddRowVector(l.Biases)
	}

	return l.output
}

func (l *FC) GetOutput() *data.Matrix {
	return l.output
}

func (l *FC) GetWeights() *data.Matrix {
	return l.Weights
}

func (l *FC) GetBiases() *data.Matrix {
	return l.Biases
}

func (l *FC) HasBiases() bool {
	return l.WithBiases
}

func (l *FC) GetInputGradients() (g *data.Matrix) {
	return l.inputs.GradsMatrix()
}

func (l *FC) GetWeightGradients() *data.Matrix {
	return l.Weights.GradsMatrix()
}
