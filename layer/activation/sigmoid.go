package activation

import (
	"github.com/atkhx/nnet/data"
)

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

type Sigmoid struct {
	inputs, output *data.Data
}

func (l *Sigmoid) Forward(inputs *data.Data) *data.Data {
	l.inputs = inputs
	l.output = l.inputs.Sigmoid()
	return l.output
}

func (l *Sigmoid) GetOutput() *data.Data {
	return l.output
}

func (l *Sigmoid) GetInputGradients() *data.Volume {
	return l.inputs.Grad
}
