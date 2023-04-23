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

func (l *Tanh) Compile(_ int, inputs *num.Data) *num.Data {
	l.inputsObj = inputs
	l.outputObj = num.New(len(inputs.GetData()))

	return l.outputObj
}

func (l *Tanh) Forward() {
	l.inputsObj.TanhTo(l.outputObj)
}
