package layer

import (
	"math"

	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/num"
)

type msaHeadWeights struct {
	KeyWeights *num.Data
	QryWeights *num.Data
	ValWeights *num.Data
}

func NewMSAHead(headSize, featuresCount int, weightK float64) msaHeadWeights {
	return msaHeadWeights{
		KeyWeights: num.NewRandNormWeighted(num.NewDims(headSize, featuresCount, 1), weightK),
		QryWeights: num.NewRandNormWeighted(num.NewDims(headSize, featuresCount, 1), weightK),
		ValWeights: num.NewRandNormWeighted(num.NewDims(headSize, featuresCount, 1), weightK),
	}
}

func NewMSAMultiHead(
	featuresCount int,
	headSize int,
	headsCount int,
	initWeights initializer.Initializer,
) *MSAMultiHead {
	return &MSAMultiHead{
		featuresCount: featuresCount,
		headSize:      headSize,
		headsCount:    headsCount,
		initWeights:   initWeights,
	}
}

type MSAMultiHead struct {
	initWeights initializer.Initializer

	featuresCount int
	headSize      int
	headsCount    int

	Heads []msaHeadWeights

	inputsObj *num.Data
	concatObj *num.Data
	forUpdate num.Nodes
}

func (l *MSAMultiHead) Compile(inputs *num.Data) *num.Data {
	l.Heads = make([]msaHeadWeights, l.headsCount)

	weightK := l.initWeights.GetNormK(len(inputs.Data))
	outputObjs := make([]*num.Data, 0, l.headsCount)

	for i := range l.Heads {
		l.Heads[i] = NewMSAHead(l.headSize, l.featuresCount, weightK)

		outputObjs = append(outputObjs, inputs.MaskedSelfAttention(
			math.Pow(float64(l.headSize), -0.5),
			l.Heads[i].KeyWeights,
			l.Heads[i].QryWeights,
			l.Heads[i].ValWeights,
		))

		l.forUpdate = append(l.forUpdate,
			l.Heads[i].KeyWeights,
			l.Heads[i].QryWeights,
			l.Heads[i].ValWeights,
		)
	}

	l.inputsObj = inputs

	if l.headsCount == 1 {
		l.concatObj = outputObjs[0]
	} else {
		l.concatObj = outputObjs[0].ConcatRows(outputObjs[1:]...)
	}

	return l.concatObj
}

func (l *MSAMultiHead) ForUpdate() num.Nodes {
	return l.forUpdate
}

func (l *MSAMultiHead) GetInputs() *num.Data {
	return l.inputsObj
}

func (l *MSAMultiHead) GetOutput() *num.Data {
	return l.concatObj
}
