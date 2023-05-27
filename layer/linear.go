package layer

import (
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/num"
)

func NewLinear(featuresCount int, initWeights initializer.Initializer) *Linear {
	return &Linear{featuresCount: featuresCount, initWeights: initWeights}
}

type Linear struct {
	initWeights   initializer.Initializer
	featuresCount int

	WeightObj *num.Data
	outputObj *num.Data
	forUpdate num.Nodes
}

func (l *Linear) Compile(inputs *num.Data) *num.Data {
	weightK := l.initWeights.GetNormK(len(inputs.Data))

	l.WeightObj = num.NewRandNormWeighted(num.NewDims(l.featuresCount, inputs.Dims.W), weightK)
	l.outputObj = inputs.MatrixMultiply(l.WeightObj)
	l.forUpdate = num.Nodes{l.WeightObj}

	return l.outputObj
}

func (l *Linear) Forward() {
	l.outputObj.Forward()
}

func (l *Linear) Backward() {
	l.outputObj.Backward()
}

func (l *Linear) ForUpdate() num.Nodes {
	return l.forUpdate
}
