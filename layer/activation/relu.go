package activation

import (
	"github.com/atkhx/nnet/data"
)

func NewReLu() *ReLu {
	return &ReLu{}
}

type ReLu struct {
	inputs, output *data.Data
}

func (l *ReLu) Forward(inputs *data.Data) *data.Data {
	l.inputs = inputs
	l.output = inputs.Relu()
	return l.output
}

func (l *ReLu) GetOutput() *data.Data {
	return l.output
}

func (l *ReLu) GetInputGradients() *data.Volume {
	return l.inputs.Grad
}
