package layer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/num"
)

func NewRMSLNorm() *RMSNorm {
	return &RMSNorm{}
}

type RMSNorm struct{}

func (l *RMSNorm) Compile(device nnet.Device, inputs *num.Data) *num.Data {
	return device.RMSNorm(inputs)
}
