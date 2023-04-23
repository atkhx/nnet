package layer

import "github.com/atkhx/nnet/num"

func NewReLu() *ReLu {
	return &ReLu{}
}

type ReLu struct {
	inputsObj *num.Data
	outputObj *num.Data
}

func (l *ReLu) Compile(_ int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	output := num.NewFloat64s(len(inputs))
	oGrads := num.NewFloat64s(len(inputs))

	l.inputsObj = num.Wrap(inputs, iGrads)
	l.outputObj = num.Wrap(output, oGrads)

	return output, oGrads
}

func (l *ReLu) Forward() {
	l.inputsObj.ReLuTo(l.outputObj)
}

func (l *ReLu) Backward() {
	l.outputObj.CalcGrad()
}

func (l *ReLu) ResetGrads() {
	l.outputObj.ResetGrad()
}
