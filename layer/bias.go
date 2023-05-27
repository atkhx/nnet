package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewBias() *Bias {
	return &Bias{}
}

type Bias struct {
	WeightObj *num.Data
	outputObj *num.Data
	forUpdate num.Nodes
}

func (l *Bias) Compile(inputs *num.Data) *num.Data {
	l.WeightObj = num.New(num.NewDims(inputs.Dims.W, inputs.Dims.H))
	l.outputObj = inputs.Add(l.WeightObj)
	l.forUpdate = num.Nodes{l.WeightObj}

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
