package layer

import "github.com/atkhx/nnet/num"

func NewDropout(prob float64) *Dropout {
	return &Dropout{Prob: prob}
}

type Dropout struct {
	Prob float64

	inputsObj *num.Data
	outputObj *num.Data
}

func (l *Dropout) Compile(inputs *num.Data) *num.Data {
	l.inputsObj = inputs
	l.outputObj = inputs.Dropout(l.Prob)
	return l.outputObj
}

func (l *Dropout) GetInputs() *num.Data {
	return l.inputsObj
}

func (l *Dropout) GetOutput() *num.Data {
	return l.outputObj
}
