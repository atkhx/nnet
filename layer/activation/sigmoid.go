package activation

import (
	"github.com/atkhx/nnet/data"
)

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

type Sigmoid struct {
	inputs, output *data.Matrix
}

func (l *Sigmoid) Forward(inputs *data.Matrix) *data.Matrix {
	l.inputs = inputs
	l.output = l.inputs.Sigmoid()
	return l.output
}

func (l *Sigmoid) GetOutput() *data.Matrix {
	return l.output
}

func (l *Sigmoid) GetInputGradients() *data.Matrix {
	return l.inputs.GradsMatrix()
}
