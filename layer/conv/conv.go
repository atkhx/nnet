package conv

import (
	"math"

	"github.com/atkhx/nnet/data"
)

func New(options ...Option) *Conv {
	layer := &Conv{}
	layer.gain = data.ReLuGain
	layer.batchSize = 1

	applyOptions(layer, defaults...)
	applyOptions(layer, options...)

	layer.Filters = data.NewRandom(
		layer.FilterSize*layer.FilterSize,
		layer.inputChannels,
		layer.FiltersCount,
	)

	if layer.gain > 0 {
		fanIn := layer.FilterSize * layer.FilterSize * layer.inputChannels * layer.batchSize
		initK := layer.gain / math.Pow(float64(fanIn), 0.5)
		layer.Filters.Data.MulScalar(initK)
	}

	layer.Biases = data.NewData(layer.FiltersCount, 1, 1)
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

	batchSize int
	gain      float64

	inputs *data.Data
	output *data.Data
}

func (l *Conv) Forward(inputs *data.Data) *data.Data {
	l.inputs = inputs
	l.output = inputs.Conv(
		l.inputWidth,
		l.inputHeight,
		l.inputChannels,
		l.FilterSize,
		l.Padding,
		l.Stride,
		l.Filters,
		l.Biases,
	)

	//fmt.Println("inputs dims", l.inputs.GetDims())
	//fmt.Println("output dims", l.output.GetDims())
	//fmt.Println("-------------------------")
	//os.Exit(1)

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
