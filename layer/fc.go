package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewFC(dims num.Dims, initWeights InitWeights) *FC {
	return &FC{dims: dims, initWeights: initWeights}
}

type FC struct {
	initWeights InitWeights

	dims num.Dims

	WeightObj *num.Data
	outputObj *num.Data
	forUpdate num.Nodes
}

func (l *FC) Compile(inputs *num.Data) *num.Data {
	weightK := l.initWeights.GetNormK(len(inputs.Data))

	l.WeightObj = num.NewRandNormWeighted(l.dims, weightK)
	l.outputObj = inputs.MatrixMultiply(l.WeightObj)
	l.forUpdate = num.Nodes{l.WeightObj}

	return l.outputObj
}

func (l *FC) Forward() {
	l.outputObj.Forward()
}

func (l *FC) Backward() {
	l.outputObj.Backward()
}

func (l *FC) ForUpdate() num.Nodes {
	return l.forUpdate
}
