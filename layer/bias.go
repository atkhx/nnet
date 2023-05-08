package layer

import "github.com/atkhx/nnet/num"

func NewBias(dims num.Dims) *Bias {
	return &Bias{
		WeightObj: num.New(dims),
	}
}

type Bias struct {
	WeightObj *num.Data
	outputObj *num.Data
}

func (l *Bias) Compile(inputs *num.Data) *num.Data {
	l.outputObj = inputs.Add(l.WeightObj)
	return l.outputObj
}

func (l *Bias) Forward() {
	l.outputObj.Forward()
}

func (l *Bias) ForUpdate() num.Nodes {
	return num.Nodes{l.WeightObj}
}
