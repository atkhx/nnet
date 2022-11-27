package activation

import (
	"math"

	"github.com/atkhx/nnet/data"
)

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

type Sigmoid struct {
	output *data.Data
	iGrads *data.Data
}

func (l *Sigmoid) InitDataSizes(w, h, d int) (int, int, int) {
	l.output = &data.Data{}
	l.output.Init3D(w, h, d)

	l.iGrads = &data.Data{}
	l.iGrads.Init3D(w, h, d)

	return w, h, d
}

func (l *Sigmoid) Forward(inputs *data.Data) *data.Data {
	output := l.output.Data
	copy(output, inputs.Data)

	for i, v := range output {
		output[i] = 1 / (1 + math.Exp(-v))
	}
	return l.output
}

func (l *Sigmoid) Backward(deltas *data.Data) *data.Data {
	output := l.output.Data
	iGrads := l.iGrads.Data

	copy(iGrads, deltas.Data)
	for i, v := range output {
		iGrads[i] *= v * (1 - v)
	}
	return l.iGrads
}

func (l *Sigmoid) GetOutput() *data.Data {
	return l.output
}

func (l *Sigmoid) GetInputGradients() *data.Data {
	return l.iGrads
}
