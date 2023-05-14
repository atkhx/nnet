package layer

import (
	"math"

	"github.com/atkhx/nnet/num"
)

// input:      [ featuresCount, contextLength, batchSize ]

// keyWeights: [ headSize, featuresCount, 1 ]
// keyObject:  [ headSize, contextLength, batchSize ]

// qryWeights: [ headSize, featuresCount, 1 ]
// qryObject:  [ headSize, contextLength, batchSize ]

// weiObject:  [ keyObject @ qryObject.T ]
//             [ headSize, contextLength, batchSize ] @ [ contextLength, headSize, batchSize ]
//          => [ contextLength, contextLength, batchSize ]

// valWeights: [ headSize, featuresCount, 1 ]
// valObject:  [ headSize, contextLength, batchSize ]

// outObject:  [ weight @ valObject ]
//             [ contextLength, contextLength, batchSize ] @ [ headSize, contextLength, batchSize ]
//          => [ headSize, contextLength, batchSize ]

func NewSAHead(
	featuresCount int,
	headSize int,
) *SAHead {
	return &SAHead{
		featuresCount: featuresCount,
		headSize:      headSize,
	}
}

type SAHead struct {
	featuresCount int
	headSize      int

	KeyWeights *num.Data
	QryWeights *num.Data
	ValWeights *num.Data

	outObject *num.Data
}

func (l *SAHead) Compile(inputs *num.Data) *num.Data {
	weightK := num.LinearGain / math.Pow(float64(len(inputs.Data)), 0.5)

	l.KeyWeights = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)
	l.QryWeights = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)
	l.ValWeights = num.NewRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)

	l.outObject = inputs.SAHead(
		l.headSize,
		l.KeyWeights,
		l.QryWeights,
		l.ValWeights,
	)

	return l.outObject
}

func (l *SAHead) Forward() {
	l.outObject.Forward()
}

func (l *SAHead) Backward() {
	l.outObject.Backward()
}

func (l *SAHead) ForUpdate() num.Nodes {
	return num.Nodes{
		l.KeyWeights,
		l.QryWeights,
		l.ValWeights,
	}
}
