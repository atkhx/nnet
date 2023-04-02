package reshape

import (
	"github.com/atkhx/nnet/data"
)

type SizeFunc func(iw, ih, id int) (ow, oh, od int)

func New(sizeFunc SizeFunc) *Reshape {
	return &Reshape{reshape: func(input *data.Data) (outMatrix *data.Data) {
		// get new sizes for Data from user defined function
		iw, ih, id := sizeFunc(input.Data.W, input.Data.H, input.Data.D)

		// todo here we can not copy data, but just change dimensions and back it on backward pass
		return input.Generate(
			data.WrapVolume(iw, ih, id, data.Copy(input.Data.Data)),
			func() {
				input.Grad.Data = data.Copy(outMatrix.Grad.Data)
			},
			input,
		)
	}}
}

type Reshape struct {
	reshape func(input *data.Data) *data.Data
}

func (l *Reshape) Forward(inputs *data.Data) *data.Data {
	return l.reshape(inputs)
}
