package activation

import (
	"github.com/atkhx/nnet/data"
)

func NewTanh() *Tanh {
	return &Tanh{}
}

type Tanh struct {
	inputs, output *data.Matrix
}

func (l *Tanh) Forward(inputs *data.Matrix) *data.Matrix {
	l.inputs = inputs
	l.output = l.inputs.Tanh()
	return l.output
}

func (l *Tanh) GetOutput() *data.Matrix {
	return l.output
}

func (l *Tanh) GetInputGradients() *data.Matrix {
	return l.inputs.GradsMatrix()
}
