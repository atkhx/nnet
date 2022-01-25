//go:generate mockgen -package=mocks -source=$GOFILE -destination=mocks/$GOFILE
package activation

import (
	"github.com/atkhx/nnet/data"
)

type ActivationFunc interface {
	Forward(v float64) float64
	Backward(v float64) float64
}

func New(f ActivationFunc) *layer {
	return &layer{Activation: f}
}

type layer struct {
	iWidth, iHeight, iDepth int

	inputs *data.Data
	output *data.Data

	gradInputs *data.Data
	Activation ActivationFunc
}

func (l *layer) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d

	l.output = &data.Data{}
	l.output.InitCube(w, h, d)

	l.gradInputs = &data.Data{}
	l.gradInputs.InitCube(w, h, d)

	return w, h, d
}

func (l *layer) Activate(inputs *data.Data) *data.Data {
	l.inputs = inputs

	for i := 0; i < len(l.inputs.Data); i++ {
		l.output.Data[i] = l.Activation.Forward(l.inputs.Data[i])
	}
	return l.output
}

func (l *layer) Backprop(deltas *data.Data) *data.Data {
	for i := 0; i < len(l.gradInputs.Data); i++ {
		l.gradInputs.Data[i] = deltas.Data[i] * l.Activation.Backward(l.output.Data[i])
	}
	return l.gradInputs
}

func (l *layer) GetOutput() *data.Data {
	return l.output
}

func (l *layer) GetInputGradients() *data.Data {
	return l.gradInputs
}
