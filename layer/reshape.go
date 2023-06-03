package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewReshape(dims num.Dims) *Reshape {
	return &Reshape{dims: dims}
}

type Reshape struct {
	dims      num.Dims
	inputsObj *num.Data
	outputObj *num.Data
}

func (l *Reshape) Compile(inputs *num.Data) *num.Data {
	l.inputsObj = inputs
	l.outputObj = inputs.Reshape(l.dims)
	return l.outputObj
}

func (l *Reshape) Forward() {
	l.outputObj.Forward()
}

func (l *Reshape) Backward() {
	l.outputObj.Backward()
}

func (l *Reshape) GetInputs() *num.Data {
	return l.inputsObj
}

func (l *Reshape) GetOutput() *num.Data {
	return l.outputObj
}
