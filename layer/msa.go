package layer

import (
	"math"

	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/num"
)

type SAHeadWeights[data any] struct {
	KeyWeights data
	QryWeights data
	ValWeights data
}

func NewMSAHead[data any](
	featuresCount int,
	headSize int,
	dropoutProb float32,
	initWeights initializer.Initializer,
) *MSAHead[data] {
	return &MSAHead[data]{
		featuresCount: featuresCount,
		headSize:      headSize,
		initWeights:   initWeights,
		dropoutProb:   dropoutProb,
	}
}

type MSAHead[data any] struct {
	initWeights initializer.Initializer

	featuresCount int
	headSize      int

	KeyWeights  data
	QryWeights  data
	ValWeights  data
	dropoutProb float32

	forUpdate []data
}

func (l *MSAHead[data]) Compile(device nnet.Device[data], inputs data) data {
	weightK := l.initWeights.GetNormK(device.GetDataLength(inputs))

	l.KeyWeights = device.NewDataRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)
	l.QryWeights = device.NewDataRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)
	l.ValWeights = device.NewDataRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)

	l.forUpdate = []data{l.KeyWeights, l.QryWeights, l.ValWeights}

	k := float32(math.Pow(float64(l.headSize), -0.5))

	keyObject := device.MatrixMultiply(inputs, l.KeyWeights)
	qryObject := device.MatrixMultiply(inputs, l.QryWeights)
	valObject := device.MatrixMultiply(inputs, l.ValWeights)

	weiObject := device.MatrixMultiply(keyObject, device.Transpose(qryObject), num.WithMatrixMultiplyAlpha(k))
	weiSoftmax := device.TriangleLowerSoftmax(weiObject)
	weiSoftmax = device.Dropout(weiSoftmax, l.dropoutProb)

	return device.MatrixMultiply(weiSoftmax, valObject)
}

func (l *MSAHead[data]) ForUpdate() []data {
	return l.forUpdate
}
