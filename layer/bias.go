package layer

import (
	"fmt"

	"github.com/atkhx/nnet/num"
)

func NewBias(dims num.Dims) *Bias {
	return &Bias{
		WeightObj: num.New(dims),
	}
}

type Bias struct {
	WeightObj *num.Data
	outputObj *num.Data
	forUpdate num.Nodes
}

func (l *Bias) Compile(inputs *num.Data) *num.Data {
	l.outputObj = inputs.Add(l.WeightObj)
	l.forUpdate = num.Nodes{l.WeightObj}

	fmt.Println("Bias\t", l.WeightObj.Dims, "out", l.outputObj.Dims)
	return l.outputObj
}

func (l *Bias) Forward() {
	l.outputObj.Forward()
}

func (l *Bias) Backward() {
	l.outputObj.Backward()
}

func (l *Bias) ForUpdate() num.Nodes {
	return l.forUpdate
}
