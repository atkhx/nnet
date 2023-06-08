package layer

import (
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/num"
)

func NewSAHead(
	featuresCount int,
	headSize int,
	initWeights initializer.Initializer,
) *SAHead {
	return &SAHead{
		featuresCount: featuresCount,
		headSize:      headSize,
		initWeights:   initWeights,
	}
}

type SAHead struct {
	initWeights initializer.Initializer

	featuresCount int
	headSize      int

	KeyWeights *num.Data
	QryWeights *num.Data
	ValWeights *num.Data

	inputsObj *num.Data
	outObject *num.Data
}

func (l *SAHead) Compile(inputs *num.Data) *num.Data {
	weightK := l.initWeights.GetNormK(len(inputs.Data))

	l.KeyWeights = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)
	l.QryWeights = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)
	l.ValWeights = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)

	l.inputsObj = inputs
	l.outObject = inputs.SAHead(
		l.headSize,
		l.KeyWeights,
		l.QryWeights,
		l.ValWeights,
	)

	return l.outObject
}

func (l *SAHead) ForUpdate() num.Nodes {
	return num.Nodes{
		l.KeyWeights,
		l.QryWeights,
		l.ValWeights,
	}
}

func (l *SAHead) GetInputs() *num.Data {
	return l.inputsObj
}

func (l *SAHead) GetOutput() *num.Data {
	return l.outObject
}
