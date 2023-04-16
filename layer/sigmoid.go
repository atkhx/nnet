package layer

import (
	"math"

	"github.com/atkhx/nnet/num"
)

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

type Sigmoid struct {
	// buffers from the previous layer
	inputs num.Float64s
	iGrads num.Float64s

	// buffers to the next layer
	output num.Float64s
	oGrads num.Float64s
}

func (l *Sigmoid) Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	l.inputs = inputs
	l.iGrads = iGrads

	l.output = make(num.Float64s, len(inputs))
	l.oGrads = make(num.Float64s, len(inputs))

	return l.output, l.oGrads
}

func (l *Sigmoid) Forward() {
	for i, v := range l.inputs {
		l.output[i] = 1.0 / (1.0 + math.Exp(-v))
	}
}

func (l *Sigmoid) Backward() {
	for i, v := range l.output {
		l.iGrads[i] += l.oGrads[i] * v * (1 - v)
	}
}

func (l *Sigmoid) ResetGrads() {
	l.oGrads.Fill(0)
}
