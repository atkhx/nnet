package softmax

import (
	"github.com/atkhx/nnet/data"
)

func New() *Softmax {
	return &Softmax{}
}

type Softmax struct {
	inputs, output *data.Data
}

func (l *Softmax) Forward(inputs *data.Data) *data.Data {
	l.inputs = inputs
	l.output = inputs.Softmax()
	return l.output
}

func (l *Softmax) GetOutput() *data.Data {
	return l.output
}

func (l *Softmax) GetInputGradients() *data.Volume {
	return l.inputs.Grad
}
