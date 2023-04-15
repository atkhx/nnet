package layer

import (
	"math"
)

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

type Sigmoid struct {
	// buffers from the previous layer
	inputs []float64
	iGrads []float64

	// buffers to the next layer
	output []float64
	oGrads []float64
}

func (l *Sigmoid) Compile(inputs, iGrads []float64) ([]float64, []float64) {
	l.inputs = inputs
	l.iGrads = iGrads

	l.output = make([]float64, len(inputs))
	l.oGrads = make([]float64, len(inputs))

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
