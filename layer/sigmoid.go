package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

type Sigmoid struct {
	outputObj *num.Data
}

func (l *Sigmoid) Compile(inputs *num.Data) *num.Data {
	l.outputObj = inputs.Sigmoid()
	return l.outputObj
}

func (l *Sigmoid) Forward() {
	l.outputObj.Forward()
}

func (l *Sigmoid) Backward() {
	l.outputObj.Backward()
}
