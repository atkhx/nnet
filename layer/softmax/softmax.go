package softmax

import (
	"math"

	"github.com/atkhx/nnet/data"
)

func New() *Softmax {
	return &Softmax{}
}

type Softmax struct {
	iWidth, iHeight, iDepth int
	oWidth, oHeight, oDepth int

	inputs *data.Data
	output *data.Data
}

func (l *Softmax) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d
	l.oWidth, l.oHeight, l.oDepth = w, h, d

	l.output = &data.Data{}
	l.output.Init3D(w, h, d)

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *Softmax) Forward(inputs *data.Data) *data.Data {
	l.inputs = inputs

	summ := 0.0
	maxv := l.inputs.GetMaxValue()

	cnt := len(l.inputs.Data)

	for i := 0; i < cnt; i++ {
		l.output.Data[i] = math.Exp(l.inputs.Data[i] - maxv)
		summ += l.output.Data[i]
	}

	for i := 0; i < cnt; i++ {
		l.output.Data[i] /= summ
	}

	return l.output
}

func (l *Softmax) GetOutput() *data.Data {
	return l.output
}

// Backward - honest implementation, which requires hones cross-entropy in loss.Classification.
func (l *Softmax) Backward(deltas *data.Data) *data.Data {
	iGrads := deltas.Copy()

	for i, o := range l.output.Data {
		iGrads.Data[i] *= o * (1 - o)
	}

	return iGrads
}

// BackwardCheat - quick implementation, which could work with cheat implementation in loss.Classification.
func (l *Softmax) BackwardCheat(deltas *data.Data) *data.Data {
	return deltas.Copy()
}
