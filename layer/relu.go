package layer

import "github.com/atkhx/nnet/num"

func NewReLu() *ReLu {
	return &ReLu{}
}

type ReLu struct {
	inputsObj *num.Data
	outputObj *num.Data
}

func (l *ReLu) Compile(_ int, inputs *num.Data) *num.Data {
	l.inputsObj = inputs
	l.outputObj = num.New(len(inputs.GetData()))

	return l.outputObj
}

func (l *ReLu) Forward() {
	l.inputsObj.ReLuTo(l.outputObj)
}
