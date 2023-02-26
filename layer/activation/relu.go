package activation

import (
	"github.com/atkhx/nnet/data"
)

func NewReLu() *ReLu {
	return &ReLu{}
}

type ReLu struct {
	inputs, output *data.Matrix
}

func (l *ReLu) Forward(inputs *data.Matrix) *data.Matrix {
	l.inputs = inputs
	l.output = inputs.Relu()
	return l.output
}

func (l *ReLu) GetOutput() *data.Matrix {
	return l.output
}

func (l *ReLu) GetInputGradients() *data.Matrix {
	return l.inputs.GradsMatrix()
}
