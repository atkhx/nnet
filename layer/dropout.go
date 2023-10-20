package layer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/num"
)

func NewDropout(prob float32) *Dropout {
	return &Dropout{dropoutProb: prob}
}

type Dropout struct {
	dropoutProb float32
}

func (l *Dropout) Compile(device nnet.Device, inputs *num.Data) *num.Data {
	return device.Dropout(inputs, l.dropoutProb)
}
