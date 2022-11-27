package activation

import (
	"github.com/atkhx/nnet/data"
)

func NewReLu() *ReLu {
	return &ReLu{}
}

type ReLu struct {
	output *data.Data
	iGrads *data.Data
}

func (l *ReLu) InitDataSizes(w, h, d int) (int, int, int) {
	l.output = &data.Data{}
	l.output.Init3D(w, h, d)

	l.iGrads = &data.Data{}
	l.iGrads.Init3D(w, h, d)

	return w, h, d
}

func (l *ReLu) Forward(inputs *data.Data) *data.Data {
	output := l.output.Data
	copy(output, inputs.Data)
	for i, ov := range output {
		if ov < 0 {
			output[i] = 0
		}
	}
	return l.output
}

func (l *ReLu) Backward(deltas *data.Data) *data.Data {
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

func (l *ReLu) GetOutput() *data.Data {
	return l.output
}

func (l *ReLu) GetInputGradients() *data.Data {
	return l.iGrads
}
