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

func (l *Tanh) Compile(_ int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	output := num.NewFloat64s(len(inputs))
	oGrads := num.NewFloat64s(len(inputs))

	l.inputsObj = num.Wrap(inputs, iGrads)
	l.outputObj = num.Wrap(output, oGrads)

	return output, oGrads
}

func (l *Tanh) Forward() {
	l.inputsObj.TanhTo(l.outputObj)
}

func (l *Tanh) Backward() {
	l.outputObj.CalcGrad()
}

func (l *Tanh) ResetGrads() {
	l.outputObj.ResetGrad()
}
