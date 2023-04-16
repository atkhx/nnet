package layer

import (
	"math"

	"github.com/atkhx/nnet/num"
)

func NewFC(size int, gain float64) *FC {
	return &FC{size: size, gain: gain}
}

type FC struct {
	size int
	gain float64

	iSize int
	bSize int

	// internal buffers (storable)
	Weights num.Float64s
	WGrads  num.Float64s

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

	weights := make(num.Float64s, l.iSize*l.size)
	weights.RandNormWeighted(weightK)

	l.Weights = weights
	l.WGrads = make(num.Float64s, l.iSize*l.size)

	l.inputs = inputs
	l.iGrads = iGrads

	l.output = make(num.Float64s, l.size*l.bSize)
	l.oGrads = make(num.Float64s, l.size*l.bSize)

	return l.output, l.oGrads
}

func (l *FC) Forward() {
	for b := 0; b < l.bSize; b++ {
		inputs := l.inputs[b*l.iSize : (b+1)*l.iSize]
		output := l.output[b*l.size : (b+1)*l.size]

		for o := 0; o < l.size; o++ {
			output[o] = num.Dot(inputs, l.Weights[o*l.iSize:(o+1)*l.iSize])
		}
	}
}

func (l *FC) Backward() {
	for b := 0; b < l.bSize; b++ {
		inputs := l.inputs[b*l.iSize : (b+1)*l.iSize]
		iGrads := l.iGrads[b*l.iSize : (b+1)*l.iSize]

		for i, delta := range l.oGrads[b*l.size : (b+1)*l.size] {
			iGrads.AddWeighted(l.Weights[i*l.iSize:(i+1)*l.iSize], delta)
			l.WGrads[i*l.iSize:(i+1)*l.iSize].AddWeighted(inputs, delta)
		}
	}
}

func (l *FC) ResetGrads() {
	l.oGrads.Fill(0)
	l.WGrads.Fill(0)
}

func (l *FC) ForUpdate() [][2]num.Float64s {
	return [][2]num.Float64s{{l.Weights, l.WGrads}}
}
