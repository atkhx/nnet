package conv

import (
	"github.com/atkhx/nnet/data"
)

func New(options ...Option) *Conv {
	layer := &Conv{}

	applyOptions(layer, defaults...)
	applyOptions(layer, options...)

	layer.Filters = data.NewRandom(
		layer.FilterSize*layer.FilterSize,
		layer.inputChannels,
		layer.FiltersCount,
	)

	layer.Biases = data.NewRandom(layer.FiltersCount, 1, 1)
	layer.Biases.Data.Fill(0)

	return layer
}

type Conv struct {
	// Data with chanCount = filtersCount
	Biases       *data.Data
	Filters      *data.Data
	FilterSize   int
	FiltersCount int
	Padding      int
	Stride       int

	inputWidth    int
	inputHeight   int
	inputChannels int

	inputs *data.Data
	output *data.Data
}

func (l *Conv) Forward(inputs *data.Data) *data.Data {
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

func (l *Conv) GetWeights() *data.Data {
	return l.Filters
}

func (l *Conv) GetOutput() *data.Data {
	return l.output
}

func (l *Conv) GetInputs() *data.Data {
	return l.inputs
}

func (l *Conv) GetInputGradients() *data.Volume {
	return l.inputs.Grad
}

func (l *Conv) GetBiases() *data.Data {
	return l.Biases
}

func (l *Conv) HasBiases() bool {
	return true
}
