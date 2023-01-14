package activation

import (
	"math"

	"github.com/atkhx/nnet/data"
)

func NewTahn() *Tahn {
	return &Tahn{}
}

type Tahn struct {
	output *data.Data
	iGrads *data.Data
}

func (l *Tahn) InitDataSizes(w, h, d int) (int, int, int) {
	l.output = &data.Data{}
	l.output.Init3D(w, h, d)

	l.iGrads = &data.Data{}
	l.iGrads.Init3D(w, h, d)

	return w, h, d
}

func (l *Tahn) Forward(inputs *data.Data) *data.Data {
	output := l.output.Data
	copy(output, inputs.Data)

	for i, v := range output {
		output[i] = math.Tanh(v)
	}
	return l.output
}

func (l *Tahn) Backward(deltas *data.Data) *data.Data {
	output := l.output.Data
	iGrads := l.iGrads.Data

	copy(iGrads, deltas.Data)
	for i, v := range output {
		iGrads[i] *= 1 - v*v
	}
	return l.iGrads
}

func (l *Tahn) GetOutput() *data.Data {
	return l.output
}

func (l *Tahn) GetInputGradients() *data.Data {
	return l.iGrads
}