package reshape

import (
	"github.com/atkhx/nnet/data"
)

type SizeFunc func(iw, ih, id int) (ow, oh, od int)

func New(sizeFunc SizeFunc) *Reshape {
	return &Reshape{reshape: func(input *data.Data) (outMatrix *data.Data) {
		return input.Reshape(sizeFunc(input.Data.W, input.Data.H, input.Data.D))
	}}
}

type Reshape struct {
	reshape func(input *data.Data) *data.Data
}

func (l *Reshape) Forward(inputs *data.Data) *data.Data {
	return l.reshape(inputs)
}
