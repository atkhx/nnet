package layer

import (
	"github.com/atkhx/nnet"
)

func NewDropout[data any](prob float32) *Dropout[data] {
	return &Dropout[data]{dropoutProb: prob}
}

type Dropout[data any] struct {
	dropoutProb float32
}

func (l *Dropout[data]) Compile(device nnet.Device[data], inputs data) data {
	return device.Dropout(inputs, l.dropoutProb)
}
