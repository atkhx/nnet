package layer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/num"
)

func NewReLu() *Activation {
	return &Activation{inputsAct: func(device nnet.Device, inputs *num.Data) *num.Data {
		return device.Relu(inputs)
	}}
}

type Activation struct {
	inputsAct func(device nnet.Device, inputs *num.Data) *num.Data
}

func (l *Activation) Compile(device nnet.Device, inputs *num.Data) *num.Data {
	return l.inputsAct(device, inputs)
}
