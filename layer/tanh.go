package layer

import (
	"math"

	"github.com/atkhx/nnet/num"
)

func NewTanh() *Tanh {
	return &Tanh{}
}

type Tanh struct {
	// buffers from the previous layer
	inputs num.Float64s
	iGrads num.Float64s

	// buffers to the next layer
	output num.Float64s
	oGrads num.Float64s
}

func (l *Tanh) Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	l.inputs = inputs
	l.iGrads = iGrads

	l.output = make(num.Float64s, len(inputs))
	l.oGrads = make(num.Float64s, len(inputs))

	return l.output, l.oGrads
}

func (l *Tanh) Forward() {
	for i, v := range l.inputs {
		l.output[i] = math.Tanh(v)
	}
}

func (l *Tanh) Backward() {
	for i, v := range l.output {
		l.iGrads[i] += l.oGrads[i] * (1 - v*v)
	}
}

func (l *Tanh) ResetGrads() {
	l.oGrads.Fill(0)
}
