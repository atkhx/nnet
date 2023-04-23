package layer

import "github.com/atkhx/nnet/num"

type Layer interface {
	Compile(bSize int, inputs *num.Data) *num.Data
	Forward()
}

type Updatable interface {
	ForUpdate() num.Nodes
}
