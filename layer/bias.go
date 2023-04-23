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
	wGrads  num.Float64s
}

func (l *Bias) Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	l.iSize = len(inputs) / bSize
	l.bSize = bSize

	l.Weights = make(num.Float64s, l.iSize)
	l.wGrads = make(num.Float64s, l.iSize)

	// Wrap to cleaver objects
	l.weightObj = num.Wrap(l.Weights, l.wGrads)

	output := make(num.Float64s, len(inputs))
	oGrads := make(num.Float64s, len(inputs))

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

func (l *Bias) ForUpdate() [][2]num.Float64s {
	return [][2]num.Float64s{{l.Weights, l.wGrads}}
}
