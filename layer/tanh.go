package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewTanh() *Tanh {
	return &Tanh{}
}

type Tanh struct {
	outputObj *num.Data
}

func (l *Tanh) Compile(inputs *num.Data) *num.Data {
	l.outputObj = inputs.Tanh()
	return l.outputObj
}

func (l *Tanh) Forward() {
	l.outputObj.Forward()
}

func (l *Tanh) Backward() {
	l.outputObj.Backward()
}
