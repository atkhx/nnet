package layer

import "github.com/atkhx/nnet/num"

func NewBias() *Bias {
	return &Bias{}
}

type Bias struct {
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

func (l *Bias) Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	l.iSize = len(inputs) / bSize
	l.bSize = bSize

	l.Weights = make(num.Float64s, l.iSize)
	l.WGrads = make(num.Float64s, l.iSize)

	l.inputs = inputs
	l.iGrads = iGrads

	l.output = make(num.Float64s, l.iSize*l.bSize)
	l.oGrads = make(num.Float64s, l.iSize*l.bSize)

	return l.output, l.oGrads
}

func (l *Bias) Forward() {
	copy(l.output, l.inputs)
	for b := 0; b < l.bSize; b++ {
		l.output[b*l.iSize : (b+1)*l.iSize].Add(l.Weights)
	}
}

func (l *Bias) Backward() {
	l.iGrads.Add(l.oGrads)

	for b := 0; b < l.bSize; b++ {
		l.WGrads.Add(l.oGrads[b*l.iSize : (b+1)*l.iSize])
	}
}

func (l *Bias) ResetGrads() {
	l.oGrads.Fill(0)
	l.WGrads.Fill(0)
}

func (l *Bias) ForUpdate() [][2]num.Float64s {
	return [][2]num.Float64s{{l.Weights, l.WGrads}}
}
