package batchnorm

import (
	"github.com/atkhx/nnet/data"
)

func New(options ...Option) *BatchNorm {
	layer := &BatchNorm{}
	applyOptions(layer, defaults...)
	applyOptions(layer, options...)

	layer.gain = data.NewData(layer.inputSize, 1, layer.inputDepth)
	layer.gain.Data.Fill(1)

	layer.bias = data.NewData(layer.inputSize, 1, layer.inputDepth)
	layer.gain.Data.Fill(0)

	return layer
}

type BatchNorm struct {
	inputSize  int
	inputDepth int

	gain *data.Data
	bias *data.Data

	inputs, output *data.Data
}

func (l *BatchNorm) Forward(inputs *data.Data) *data.Data {
	l.inputs = inputs
	l.output = inputs.
		SubRowVector(inputs.ColMean()).
		DivRowVector(inputs.ColStd()).
		MulRowVector(l.gain).
		AddRowVector(l.bias)

	return l.output
}

func (l *BatchNorm) shortCalc(inputs *data.Data) {

}

func (l *BatchNorm) GetOutput() *data.Data {
	return l.output
}

func (l *BatchNorm) GetWeights() *data.Data {
	return l.gain
}

func (l *BatchNorm) GetBiases() *data.Data {
	return l.bias
}

func (l *BatchNorm) HasBiases() bool {
	return true
}

func (l *BatchNorm) GetInputGradients() (g *data.Volume) {
	return l.inputs.Grad
}
