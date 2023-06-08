package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

type Sigmoid struct {
	inputsObj *num.Data
	outputObj *num.Data
}

func (l *Sigmoid) Compile(inputs *num.Data) *num.Data {
	l.inputsObj = inputs
	l.outputObj = inputs.Sigmoid()
	return l.outputObj
}

func (l *Sigmoid) GetInputs() *num.Data {
	return l.inputsObj
}

func (l *Sigmoid) GetOutput() *num.Data {
	return l.outputObj
}
