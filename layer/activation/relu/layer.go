package relu

import (
	"github.com/atkhx/nnet/data"
)

func New() *Layer {
	return &Layer{}
}

type Layer struct {
	output *data.Data
	iGrads *data.Data
}

func (l *Layer) InitDataSizes(w, h, d int) (int, int, int) {
	l.output = &data.Data{}
	l.output.InitCube(w, h, d)

	l.iGrads = &data.Data{}
	l.iGrads.InitCube(w, h, d)

	return w, h, d
}

func (l *Layer) Activate(inputs *data.Data) *data.Data {
	output := l.output.Data
	copy(output, inputs.Data)
	for i, ov := range output {
		if ov < 0 {
			output[i] = 0
		}
	}
	return l.output
}

func (l *Layer) Backprop(deltas *data.Data) *data.Data {
	output := l.output.Data
	iGrads := l.iGrads.Data

	copy(iGrads, deltas.Data)
	for i, ov := range output {
		if ov <= 0 {
			iGrads[i] = 0
		}
	}
	return l.iGrads
}

func (l *Layer) GetOutput() *data.Data {
	return l.output
}

func (l *Layer) GetInputGradients() *data.Data {
	return l.iGrads
}
