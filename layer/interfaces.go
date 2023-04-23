package layer

import "github.com/atkhx/nnet/num"

type Layer interface {
	Forward()
	Backward()

	Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s)
}

type Updatable interface {
	ForUpdate() num.Nodes
}

type WithGrads interface {
	ResetGrads()
}
