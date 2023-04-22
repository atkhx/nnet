package layer

import (
	"math"

	"github.com/atkhx/nnet/num"
)

func NewFC(size int, gain float64) *FC {
	return &FC{oSize: size, gain: gain}
}

type FC struct {
	oSize int
	gain  float64

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

func (l *FC) Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	l.iSize = len(inputs) / bSize
	l.bSize = bSize

	weightK := 1.0
	if l.gain > 0 {
		fanIn := l.iSize * l.bSize
		weightK = l.gain / math.Pow(float64(fanIn), 0.5)
	}

	weights := make(num.Float64s, l.iSize*l.oSize)
	weights.RandNormWeighted(weightK)

	l.Weights = weights
	l.wGrads = make(num.Float64s, l.iSize*l.oSize)

	l.inputs = inputs
	l.iGrads = iGrads

	l.output = make(num.Float64s, l.oSize*l.bSize)
	l.oGrads = make(num.Float64s, l.oSize*l.bSize)

	l.inputsObj = num.Wrap(l.inputs, l.iGrads)
	l.outputObj = num.Wrap(l.output, l.oGrads)
	l.weightObj = num.Wrap(l.Weights, l.wGrads)

	return l.output, l.oGrads
}

func (l *FC) Forward() {
	l.inputsObj.DotTo(l.outputObj, l.weightObj, l.bSize)
}

func (l *FC) Backward() {
	l.outputObj.CalcGrad()
}

func (l *FC) ResetGrads() {
	l.oGrads.Fill(0)
	l.wGrads.Fill(0)
}

func (l *FC) ForUpdate() [][2]num.Float64s {
	return [][2]num.Float64s{{l.Weights, l.wGrads}}
}
