package layer

import (
	"sync"

	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/num"
)

func NewSAMultiHead(
	featuresCount int,
	headSize int,
	headsCount int,
	initWeights initializer.Initializer,
) *SAMultiHead {
	return &SAMultiHead{
		featuresCount: featuresCount,
		headSize:      headSize,
		headsCount:    headsCount,
		initWeights:   initWeights,
	}
}

type SAHeadParams struct {
	KeyWeights *num.Data
	QryWeights *num.Data
	ValWeights *num.Data
	outputObj  *num.Data
}

type SAMultiHead struct {
	initWeights initializer.Initializer

	featuresCount int
	headSize      int
	headsCount    int

	Heads []SAHeadParams

	inputsObj *num.Data
	concatObj *num.Data
	forUpdate num.Nodes

	wg sync.WaitGroup
}

func (l *SAMultiHead) Compile(inputs *num.Data) *num.Data {
	weightK := l.initWeights.GetNormK(len(inputs.Data))

	l.Heads = make([]SAHeadParams, l.headsCount)

	outputObjs := make([]*num.Data, 0, l.headsCount)

	for i := 0; i < l.headsCount; i++ {
		l.Heads[i].KeyWeights = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)
		l.Heads[i].QryWeights = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)
		l.Heads[i].ValWeights = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)

		l.Heads[i].outputObj = inputs.SAHead(
			l.headSize,
			l.Heads[i].KeyWeights,
			l.Heads[i].QryWeights,
			l.Heads[i].ValWeights,
		)

		//l.Heads[i].KeyWeights = num.NewRandNormWeighted(num.NewDims(l.featuresCount, l.headSize, 1), weightK)
		//l.Heads[i].QryWeights = num.NewRandNormWeighted(num.NewDims(l.featuresCount, l.headSize, 1), weightK)
		//l.Heads[i].ValWeights = num.NewRandNormWeighted(num.NewDims(l.featuresCount, l.headSize, 1), weightK)
		//
		//l.Heads[i].outputObj = inputs.SAHeadTransposed(
		//	l.headSize,
		//	l.Heads[i].KeyWeights,
		//	l.Heads[i].QryWeights,
		//	l.Heads[i].ValWeights,
		//)

		outputObjs = append(outputObjs, l.Heads[i].outputObj)

		l.forUpdate = append(l.forUpdate,
			l.Heads[i].KeyWeights,
			l.Heads[i].QryWeights,
			l.Heads[i].ValWeights,
		)
	}

	l.inputsObj = inputs
	l.concatObj = outputObjs[0].ConcatRows(outputObjs[1:]...)

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
