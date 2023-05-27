package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewReshape(dims num.Dims) *Reshape {
	return &Reshape{dims: dims}
}

type Reshape struct {
	dims      num.Dims
	outputObj *num.Data
}

func (l *Reshape) Compile(inputs *num.Data) *num.Data {
	l.outputObj = inputs.Reshape(l.dims)
	return l.outputObj
}

func (l *Reshape) Forward() {
	l.outputObj.Forward()
}

func (l *Reshape) Backward() {
	l.outputObj.Backward()
}
