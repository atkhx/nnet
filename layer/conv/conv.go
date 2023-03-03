package conv

import (
	"github.com/atkhx/nnet/data"
)

func New(options ...Option) *Conv {
	layer := &Conv{}

	applyOptions(layer, defaults...)
	applyOptions(layer, options...)

	layer.Filters = data.NewMatrixRandom(
		layer.FilterSize*layer.FilterSize,
		layer.inputChannels,
		layer.FiltersCount,
	)

	layer.Biases = data.NewMatrixRandom(layer.FiltersCount, 1, 1)

	for i := range layer.Biases.Data {
		layer.Biases.Data[i] = 0
	}

	return layer
}

type Conv struct {
	// Matrix with chanCount = filtersCount
	Biases       *data.Matrix
	Filters      *data.Matrix
	FilterSize   int
	FiltersCount int
	Padding      int
	Stride       int

	inputWidth    int
	inputHeight   int
	inputChannels int

	inputs *data.Matrix
	output *data.Matrix
}

func (l *Conv) Forward(inputs *data.Matrix) *data.Matrix {
	l.inputs = inputs
	l.output = inputs.Conv(
		l.inputWidth,
		l.inputHeight,
		l.FilterSize,
		l.Padding,
		l.Stride,
		l.Filters,
		l.Biases,
	)

	return l.output
}

func (l *Conv) GetWeights() *data.Matrix {
	return l.Filters
}

func (l *Conv) GetOutput() *data.Matrix {
	return l.output
}

func (l *Conv) GetInputs() *data.Matrix {
	return l.inputs
}

func (l *Conv) GetInputGradients() *data.Matrix {
	return l.inputs.GradsMatrix()
}

func (l *Conv) GetWeightGradients() *data.Matrix {
	return l.Filters.GradsMatrix()
}

func (l *Conv) GetBiases() *data.Matrix {
	return l.Biases
}

func (l *Conv) HasBiases() bool {
	return true
}
