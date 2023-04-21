package layer

import (
	"math"

	"github.com/atkhx/nnet/num"
)

func NewConv(
	imageSize int,
	imageDepth int,
	filterSize int,
	filtersCount int,
	padding int,
	gain float64,
) *Conv {
	return &Conv{
		iW: imageSize,
		iH: imageSize,
		iD: imageDepth,

		filterSize:   filterSize,
		filtersCount: filtersCount,

		padding: padding,
		stride:  1,

		gain: gain,
	}
}

type Conv struct {
	iW, iH, iD int // inputs params
	oW, oH, oD int // output params

	filterSize   int
	filtersCount int

	padding int
	stride  int

	gain float64

	iSize int
	bSize int

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

func (l *Conv) Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	l.iSize = len(inputs) / bSize
	l.bSize = bSize

	l.oW, l.oH = num.CalcConvOutputSize(l.iW, l.iH, l.filterSize, l.filterSize, l.padding, l.stride)
	l.oD = l.filtersCount

	wCube := l.filterSize * l.filterSize * l.iD
	oCube := l.oW * l.oH * l.oD

	weightK := 1.0
	if l.gain > 0 {
		fanIn := wCube * l.bSize
		weightK = l.gain / math.Pow(float64(fanIn), 0.5)
	}

	weights := make(num.Float64s, wCube*l.filtersCount)
	weights.RandNormWeighted(weightK)

	l.Weights = weights
	l.wGrads = make(num.Float64s, wCube*l.filtersCount)

	l.inputs = inputs
	l.iGrads = iGrads

	l.output = make(num.Float64s, l.bSize*oCube)
	l.oGrads = make(num.Float64s, l.bSize*oCube)

	return l.output, l.oGrads
}

func (l *Conv) Forward() {

}

func (l *Conv) Backward() {

}

func (l *Conv) ResetGrads() {
	l.oGrads.Fill(0)
	l.wGrads.Fill(0)
}

func (l *Conv) ForUpdate() [][2]num.Float64s {
	return [][2]num.Float64s{{l.Weights, l.wGrads}}
}
