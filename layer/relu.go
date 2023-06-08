package layer

import "github.com/atkhx/nnet/num"

func NewReLu() *ReLu {
	return &ReLu{}
}

type ReLu struct {
	inputsObj *num.Data
	outputObj *num.Data
}

func (l *ReLu) Compile(inputs *num.Data) *num.Data {
	l.inputsObj = inputs
	l.outputObj = inputs.Relu()
	return l.outputObj
}

func (l *ReLu) GetInputs() *num.Data {
	return l.inputsObj
}

func (l *ReLu) GetOutput() *num.Data {
	return l.outputObj
}
