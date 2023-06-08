package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewTanh() *Tanh {
	return &Tanh{}
}

type Tanh struct {
	inputsObj *num.Data
	outputObj *num.Data
}

func (l *Tanh) Compile(inputs *num.Data) *num.Data {
	l.inputsObj = inputs
	l.outputObj = inputs.Tanh()
	return l.outputObj
}

func (l *Tanh) GetInputs() *num.Data {
	return l.inputsObj
}

func (l *Tanh) GetOutput() *num.Data {
	return l.outputObj
}
