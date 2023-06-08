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
	BiasesObj *num.Data

	inputsObj *num.Data
	outputObj *num.Data
	forUpdate num.Nodes
}

func (l *Linear) Compile(inputs *num.Data) *num.Data {
	weightK := l.initWeights.GetNormK(len(inputs.Data))

	l.WeightObj = num.NewRandNormWeighted(num.NewDims(l.featuresCount, inputs.Dims.W), weightK)
	l.BiasesObj = num.New(num.NewDims(l.featuresCount))

	l.inputsObj = inputs
	l.outputObj = inputs.MatrixMultiply(l.WeightObj).Add(l.BiasesObj)
	l.forUpdate = num.Nodes{l.WeightObj, l.BiasesObj}

	return l.outputObj
}

func (l *Linear) ForUpdate() num.Nodes {
	return l.forUpdate
}

func (l *Linear) GetInputs() *num.Data {
	return l.inputsObj
}

func (l *Linear) GetOutput() *num.Data {
	return l.outputObj
}
