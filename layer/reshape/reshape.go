package reshape

import (
	"github.com/atkhx/nnet/data"
)

func New(reshape func(input *data.Data) *data.Data) *Flat {
	return &Flat{reshape: reshape}
}

type Flat struct {
	reshape func(input *data.Data) *data.Data
}

func (l *Flat) Forward(inputs *data.Data) *data.Data {
	return l.reshape(inputs)
}
