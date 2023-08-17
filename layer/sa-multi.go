package layer

import (
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/num"
)

func NewSAMultiHead(
	featuresCount int,
	headSize int,
	headsCount int,
	dropoutProb float64,
	initWeights initializer.Initializer,
) *SAMultiHead {
	return &SAMultiHead{
		featuresCount: featuresCount,
		headSize:      headSize,
		headsCount:    headsCount,
		initWeights:   initWeights,
		DropoutProb:   dropoutProb,
	}
}

type SAMultiHead struct {
	initWeights initializer.Initializer

	featuresCount int
	headSize      int
	headsCount    int

	HeadWeights []num.SAHeadWeights
	headObjects []*num.Data
	DropoutProb float64

	inputsObj *num.Data
	concatObj *num.Data
	forUpdate num.Nodes
}

func (l *SAMultiHead) Compile(inputs *num.Data) *num.Data {
	l.HeadWeights = make([]num.SAHeadWeights, 0, l.headsCount)
	l.headObjects = make([]*num.Data, 0, l.headsCount)

	weightK := l.initWeights.GetNormK(len(inputs.Data))

	for i := 0; i < l.headsCount; i++ {
		l.HeadWeights = append(l.HeadWeights, num.NewSAHeadWeights(l.headSize, l.featuresCount, weightK))

		l.forUpdate = append(l.forUpdate,
			l.HeadWeights[i].KeyWeights,
			l.HeadWeights[i].QryWeights,
			l.HeadWeights[i].ValWeights,
		)

		l.headObjects = append(l.headObjects, inputs.SA(l.DropoutProb, l.HeadWeights[i]))
	}

	l.inputsObj = inputs

	if l.headsCount == 1 {
		l.concatObj = l.headObjects[0]
	} else {
		l.concatObj = l.headObjects[0].ConcatRows(l.headObjects[1:]...)
	}

	return l.concatObj
}

func (l *SAMultiHead) ForUpdate() num.Nodes {
	return l.forUpdate
}

func (l *SAMultiHead) GetInputs() *num.Data {
	return l.inputsObj
}

func (l *SAMultiHead) GetOutput() *num.Data {
	return l.concatObj
}
