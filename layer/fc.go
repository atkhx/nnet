package layer

import (
	"math/rand"

	"github.com/atkhx/nnet/num"
)

func NewFC(size int) *FC {
	return &FC{size: size}
}

type FC struct {
	size int

	// internal buffers
	weights []float64
	wGrads  []float64

	// buffers from the previous layer
	inputs []float64
	iGrads []float64

	// buffers to the next layer
	output []float64
	oGrads []float64
}

func (l *FC) Compile(inputs, iGrads []float64) ([]float64, []float64) {
	weights := make([]float64, len(inputs)*l.size)
	for i := range weights {
		weights[i] = rand.NormFloat64()
	}

	l.weights = weights
	l.wGrads = make([]float64, len(inputs)*l.size)

	l.inputs = inputs
	l.iGrads = iGrads

	l.output = make([]float64, l.size)
	l.oGrads = make([]float64, l.size)

	return l.output, l.oGrads
}

func (l *FC) Forward() {
	for o := 0; o < l.size; o++ {
		l.output[o] = num.Dot(l.inputs, l.weights[o*len(l.inputs):(o+1)*len(l.inputs)])
	}
}

func (l *FC) Backward() {
	for i, delta := range l.oGrads {
		weights := l.weights[i*len(l.inputs) : (i+1)*len(l.inputs)]
		for j, iv := range weights {
			l.iGrads[j] += delta * iv
		}

		wGrads := l.wGrads[i*len(l.inputs) : (i+1)*len(l.inputs)]
		for j, iv := range l.inputs {
			wGrads[j] += delta * iv
		}
	}
}

func (l *FC) ResetGrads() {
	for i := range l.wGrads {
		l.wGrads[i] = 0
	}

	for i := range l.oGrads {
		l.oGrads[i] = 0
	}
}

func (l *FC) ForUpdate() [][2][]float64 {
	return [][2][]float64{
		{l.weights, l.wGrads},
	}
}
