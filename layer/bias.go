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

	// buffers from the previous layer
	inputs num.Float64s
	iGrads num.Float64s

	// buffers to the next layer
	output num.Float64s
	oGrads num.Float64s
}

func (l *Bias) Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	l.iSize = len(inputs) / bSize
	l.bSize = bSize

	l.Weights = make(num.Float64s, l.iSize)
	l.wGrads = make(num.Float64s, l.iSize)

	l.inputs = inputs
	l.iGrads = iGrads

	l.output = make(num.Float64s, l.iSize*l.bSize)
	l.oGrads = make(num.Float64s, l.iSize*l.bSize)

	// Wrap to cleaver objects

	l.weightObj = num.Wrap(l.Weights, l.wGrads)
	l.inputsObj = num.Wrap(l.inputs, l.iGrads)
	l.outputObj = num.Wrap(l.output, l.oGrads)

	return l.output, l.oGrads
}

func (l *Bias) Forward() {
	l.inputsObj.AddTo(l.outputObj, l.weightObj)
}

func (l *Bias) Backward() {
	l.outputObj.CalcGrad()
}

func (l *Bias) ResetGrads() {
	l.oGrads.Fill(0)
	l.wGrads.Fill(0)
}

func (l *Bias) ForUpdate() [][2]num.Float64s {
	return [][2]num.Float64s{{l.Weights, l.wGrads}}
}
