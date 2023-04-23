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

func (l *Sigmoid) Compile(_ int, inputs *num.Data) *num.Data {
	l.inputsObj = inputs
	l.outputObj = num.New(len(inputs.GetData()))

	return l.outputObj
}

func (l *Sigmoid) Forward() {
	l.inputsObj.SigmoidTo(l.outputObj)
}
