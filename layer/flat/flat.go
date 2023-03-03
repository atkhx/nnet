package flat

import (
	"github.com/atkhx/nnet/data"
)

func New() *Flat {
	layer := &Flat{}
	return layer
}

type Flat struct {
	inputs *data.Matrix
	output *data.Matrix
}

func (l *Flat) Forward(inputs *data.Matrix) *data.Matrix {
	//fmt.Println("-------")
	//fmt.Println("conv layer")
	//fmt.Println("inputs", inputs.GetDims())
	l.inputs = inputs
	l.output = inputs.Flat()
	//fmt.Println("output", l.output.GetDims())
	return l.output
}
