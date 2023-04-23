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

func (l *Sigmoid) Compile(_ int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	output := make(num.Float64s, len(inputs))
	oGrads := make(num.Float64s, len(inputs))

	l.inputsObj = num.Wrap(inputs, iGrads)
	l.outputObj = num.Wrap(output, oGrads)

	return output, oGrads
}

func (l *Sigmoid) Forward() {
	l.inputsObj.SigmoidTo(l.outputObj)
}

func (l *Sigmoid) Backward() {
	l.outputObj.CalcGrad()
}

func (l *Sigmoid) ResetGrads() {
	l.outputObj.ResetGrad()
}
