package layer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/num"
)

func NewReshape[data any](dims num.Dims) *Reshape[data] {
	return &Reshape[data]{dims: dims}
}

type Reshape[data any] struct {
	dims num.Dims
}

func (l *Reshape[data]) Compile(device nnet.Device[data], inputs data) data {
	return device.Reshape(inputs, l.dims)
}
