package softmax

import (
	"math"

	"github.com/atkhx/nnet/data"
)

func New(options ...Option) *Layer {
	layer := &Layer{}
	applyOptions(layer, defaults...)
	applyOptions(layer, options...)

	return layer
}

type Layer struct {
	iWidth, iHeight, iDepth int
	oWidth, oHeight, oDepth int

	inputs *data.Data
	output *data.Data
}

func (l *Layer) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d
	l.oWidth, l.oHeight, l.oDepth = w, h, d

	l.output = &data.Data{}
	l.output.InitCube(w, h, d)

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *Layer) Activate(inputs *data.Data) *data.Data {
	l.inputs = inputs

	summ := 0.0
	maxv := l.inputs.GetMaxValue()

	cnt := len(l.inputs.Data)

	for i := 0; i < cnt; i++ {
		l.output.Data[i] = math.Exp(l.inputs.Data[i] - maxv)
		summ += l.output.Data[i]
	}

	for i := 0; i < cnt; i++ {
		l.output.Data[i] /= summ
	}

	return l.output
}

func (l *Layer) GetOutput() *data.Data {
	return l.output
}

func (l *Layer) Backprop(deltas *data.Data) *data.Data {
	return deltas.Copy()
}
