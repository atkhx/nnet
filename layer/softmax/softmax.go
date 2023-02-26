package softmax

import (
	"github.com/atkhx/nnet/data"
)

func New() *Softmax {
	return &Softmax{}
}

type Softmax struct {
	inputs, output *data.Matrix
}

func (l *Softmax) Forward(inputs *data.Matrix) *data.Matrix {
	l.inputs = inputs
	l.output = inputs.SoftmaxRows()
	return l.output
}

func (l *Softmax) GetOutput() *data.Matrix {
	return l.output
}

func (l *Softmax) GetInputGradients() *data.Matrix {
	return l.inputs.GradsMatrix()
}
