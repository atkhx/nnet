package layer

import (
	"math"
)

func NewSigmoid(inputs, iGrads []float64) *Sigmoid {
	return &Sigmoid{
		inputs: inputs,
		iGrads: iGrads,

		output: make([]float64, len(inputs)),
		oGrads: make([]float64, len(inputs)),
	}
}

type Sigmoid struct {
	// buffers from the previous layer
	inputs []float64
	iGrads []float64

	// buffers to the next layer
	output []float64
	oGrads []float64
}

func (l *Sigmoid) Buffers() (output, oGrads []float64) {
	return l.output, l.oGrads
}

func (l *Sigmoid) Forward() {
	for i, v := range l.inputs {
		l.output[i] = 1 / (1 + math.Exp(-v))
	}
}

func (l *Sigmoid) Backward() {
	for i, v := range l.output {
		l.iGrads[i] += l.oGrads[i] * v * (1 - v)
	}
}

func (l *Sigmoid) ResetGrads() {
	for i := range l.oGrads {
		l.oGrads[i] = 0
	}
}
