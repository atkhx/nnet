package layer

import (
	"math"

	"github.com/atkhx/nnet/num"
)

func NewSAMultiHead(
	featuresCount int,
	headSize int,
) *SAMultiHead {
	return &SAMultiHead{
		featuresCount: featuresCount,
		headSize:      headSize / 2,
	}
}

type SAMultiHead struct {
	featuresCount int
	headSize      int

	KeyWeights1 *num.Data
	QryWeights1 *num.Data
	ValWeights1 *num.Data

	outObject1 *num.Data

	KeyWeights2 *num.Data
	QryWeights2 *num.Data
	ValWeights2 *num.Data

	outObject2 *num.Data

	concatObj *num.Data
}

func (l *SAMultiHead) Compile(inputs *num.Data) *num.Data {
	weightK := num.LinearGain / math.Pow(float64(len(inputs.Data)), 0.5)

	l.KeyWeights1 = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)
	l.QryWeights1 = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)
	l.ValWeights1 = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)

	l.KeyWeights2 = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)
	l.QryWeights2 = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)
	l.ValWeights2 = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)

	l.outObject1 = inputs.SAHead(
		l.headSize,
		l.KeyWeights1,
		l.QryWeights1,
		l.ValWeights1,
	)

	l.outObject2 = inputs.SAHead(
		l.headSize,
		l.KeyWeights2,
		l.QryWeights2,
		l.ValWeights2,
	)

	l.concatObj = l.outObject1.ConcatRows(l.outObject2)

	return l.concatObj
}

func (l *SAMultiHead) Forward() {
	l.outObject1.Forward()
	l.outObject2.Forward()
	l.concatObj.Forward()
}

func (l *SAMultiHead) ForUpdate() num.Nodes {
	return num.Nodes{
		l.KeyWeights1,
		l.QryWeights1,
		l.ValWeights1,

		l.KeyWeights2,
		l.QryWeights2,
		l.ValWeights2,
	}
}
