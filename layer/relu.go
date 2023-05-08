package layer

import "github.com/atkhx/nnet/num"

func NewReLu() *ReLu {
	return &ReLu{}
}

type ReLu struct {
	outputObj *num.Data
}

func (l *ReLu) Compile(inputs *num.Data) *num.Data {
	l.outputObj = inputs.Relu()
	return l.outputObj
}

func (l *ReLu) Forward() {
	l.outputObj.Forward()
}
