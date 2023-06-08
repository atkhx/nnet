package layer

import "github.com/atkhx/nnet/num"

type Layer interface {
	Compile(inputs *num.Data) *num.Data
	GetInputs() *num.Data
	GetOutput() *num.Data
}

type Updatable interface {
	ForUpdate() num.Nodes
}
