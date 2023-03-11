package activation

import (
	"github.com/atkhx/nnet/data"
)

func NewTanh() *Tanh {
	return &Tanh{}
}

type Tanh struct {
	inputs, output *data.Data
}

func (l *Tanh) Forward(inputs *data.Data) *data.Data {
	l.inputs = inputs
	l.output = l.inputs.Tanh()
	return l.output
}

func (l *Tanh) GetOutput() *data.Data {
	return l.output
}

func (l *Tanh) GetInputGradients() *data.Volume {
	return l.inputs.Grad
}
