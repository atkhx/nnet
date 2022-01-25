package softmax

import (
	"github.com/atkhx/nnet/data"
	"math"
)

func New(options ...Option) *layer {
	layer := &layer{}
	defaults(layer)

	for _, opt := range options {
		opt(layer)
	}

	return layer
}

type layer struct {
	IWidth, IHeight, IDepth int
	OWidth, OHeight, ODepth int

	inputs *data.Data
	output *data.Data

	debug bool
}

func (l *layer) InitDataSizes(w, h, d int) (int, int, int) {
	l.IWidth, l.IHeight, l.IDepth = w, h, d
	l.OWidth, l.OHeight, l.ODepth = w, h, d

	l.output = &data.Data{}
	l.output.InitCube(w, h, d)

	return l.OWidth, l.OHeight, l.ODepth
}

func (l *layer) Activate(inputs *data.Data) *data.Data {
	l.inputs = inputs

	summ := 0.0
	maxv := 0.0

	cnt := len(l.inputs.Data)

	for i := 0; i < cnt; i++ {
		if i == 0 || maxv < l.inputs.Data[i] {
			maxv = l.inputs.Data[i]
		}
	}

	for i := 0; i < cnt; i++ {
		l.output.Data[i] = math.Exp(l.inputs.Data[i] - maxv)
		summ += l.output.Data[i]
	}

	for i := 0; i < cnt; i++ {
		l.output.Data[i] /= summ
	}

	return l.output
}

func (l *layer) GetOutput() *data.Data {
	return l.output
}

func (l *layer) Backprop(deltas *data.Data) *data.Data {
	return deltas.Copy()
}
