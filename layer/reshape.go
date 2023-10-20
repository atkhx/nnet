package layer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/num"
)

func NewReshape(dims num.Dims) *Reshape {
	return &Reshape{dims: dims}
}

type Reshape struct {
	dims num.Dims
}

func (l *Reshape) Compile(device nnet.Device, inputs *num.Data) *num.Data {
	return device.Reshape(inputs, l.dims)
}
