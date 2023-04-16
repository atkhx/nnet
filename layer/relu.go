package layer

import "github.com/atkhx/nnet/num"

func NewReLu() *ReLu {
	return &ReLu{}
}

type ReLu struct {
	// buffers from the previous layer
	inputs num.Float64s
	iGrads num.Float64s

	// buffers to the next layer
	output num.Float64s
	oGrads num.Float64s
}

func (l *ReLu) Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	l.inputs = inputs
	l.iGrads = iGrads

	l.output = make(num.Float64s, len(inputs))
	l.oGrads = make(num.Float64s, len(inputs))

	return l.output, l.oGrads
}

func (l *ReLu) Forward() {
	for i, v := range l.inputs {
		if v > 0 {
			l.output[i] = v
		} else {
			l.output[i] = 0
		}
	}
}

func (l *ReLu) Backward() {
	for i, v := range l.output {
		if v > 0 {
			l.iGrads[i] += l.oGrads[i]
		}
	}
}

func (l *ReLu) ResetGrads() {
	l.oGrads.Fill(0)
}
