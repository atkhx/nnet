package layer

import "github.com/atkhx/nnet/num"

func NewBias() *Bias {
	return &Bias{}
}

type Bias struct {
	iSize int
	bSize int

	// clever objects
	weightObj *num.Data
	inputsObj *num.Data
	outputObj *num.Data

	// internal buffers
	Weights num.Float64s // (storable)
}

func (l *Bias) Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	l.iSize = len(inputs) / bSize
	l.bSize = bSize

	l.Weights = num.NewFloat64s(l.iSize)
	l.weightObj = num.Wrap(l.Weights, num.NewFloat64s(l.iSize))

	output := num.NewFloat64s(len(inputs))
	oGrads := num.NewFloat64s(len(inputs))

	l.inputsObj = num.Wrap(inputs, iGrads)
	l.outputObj = num.Wrap(output, oGrads)

	return output, oGrads
}

func (l *Bias) Forward() {
	l.inputsObj.AddTo(l.outputObj, l.weightObj)
}

func (l *Bias) Backward() {
	l.outputObj.CalcGrad()
}

func (l *Bias) ResetGrads() {
	l.outputObj.ResetGrad()
	l.weightObj.ResetGrad()
}

func (l *Bias) ForUpdate() num.Nodes {
	return num.Nodes{l.weightObj}
}
