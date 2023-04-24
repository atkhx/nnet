package layer

import (
	"math"

	"github.com/atkhx/nnet/num"
)

func NewFC(size int, gain float64) *FC {
	return &FC{oSize: size, gain: gain}
}

type FC struct {
	gain float64

	iSize int
	bSize int
	oSize int

	// clever objects
	weightObj *num.Data
	inputsObj *num.Data
	outputObj *num.Data

	Weights num.Float64s // (storable)
}

func (l *FC) Compile(bSize int, inputs *num.Data) *num.Data {
	inputsLen := len(inputs.GetData())

	l.iSize = inputsLen / bSize
	l.bSize = bSize

	weightK := 1.0
	if l.gain > 0 {
		fanIn := l.iSize * l.bSize
		weightK = l.gain / math.Pow(float64(fanIn), 0.5)
	}

	l.Weights = num.NewFloat64sRandNormWeighted(l.iSize*l.oSize, weightK)
	l.weightObj = num.Wrap(l.Weights, num.NewFloat64s(l.iSize*l.oSize))

	l.inputsObj = inputs
	l.outputObj = num.New(l.oSize * l.bSize)

	return l.outputObj
}

func (l *FC) Forward() {
	l.inputsObj.DotTo(l.outputObj, l.weightObj, l.bSize)
}

func (l *FC) ForUpdate() num.Nodes {
	return num.Nodes{l.weightObj}
}
