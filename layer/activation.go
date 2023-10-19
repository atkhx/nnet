package layer

import (
	"github.com/atkhx/nnet"
)

func NewReLu[data any]() *Activation[data] {
	return &Activation[data]{inputsAct: func(device nnet.Device[data], inputs data) data {
		return device.Relu(inputs)
	}}
}

type Activation[data any] struct {
	inputsAct func(device nnet.Device[data], inputs data) data
}

func (l *Activation[data]) Compile(device nnet.Device[data], inputs data) data {
	return l.inputsAct(device, inputs)
}
