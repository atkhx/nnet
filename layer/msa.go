package layer

import (
	"math"

	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/num"
)

type SAHeadWeights struct {
	KeyWeights *num.Data
	QryWeights *num.Data
	ValWeights *num.Data
}

func NewMSAHead(
	featuresCount int,
	headSize int,
	dropoutProb float32,
	initWeights initializer.Initializer,
) *MSAHead {
	return &MSAHead{
		featuresCount: featuresCount,
		headSize:      headSize,
		initWeights:   initWeights,
		dropoutProb:   dropoutProb,
	}
}

type MSAHead struct {
	initWeights initializer.Initializer

	featuresCount int
	headSize      int

	KeyWeights  *num.Data
	QryWeights  *num.Data
	ValWeights  *num.Data
	dropoutProb float32

	forUpdate []*num.Data
}

func (l *MSAHead) Compile(device nnet.Device, inputs *num.Data) *num.Data {
	weightK := l.initWeights.GetNormK(device.GetDataLength(inputs))

	l.KeyWeights = device.NewDataRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)
	l.QryWeights = device.NewDataRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)
	l.ValWeights = device.NewDataRandNormWeighted(num.NewDims(l.headSize, l.featuresCount, 1), weightK)

	l.forUpdate = []*num.Data{l.KeyWeights, l.QryWeights, l.ValWeights}

	k := float32(math.Pow(float64(l.headSize), -0.5))

	keyObject := device.MatrixMultiply3D(inputs, l.KeyWeights, 1)
	qryObject := device.MatrixMultiply3D(inputs, l.QryWeights, 1)
	valObject := device.MatrixMultiply3D(inputs, l.ValWeights, 1)

	weiObject := device.MatrixMultiply3D(keyObject, device.Transpose(qryObject), k)
	weiSoftmax := device.TriangleLowerSoftmax(weiObject)
	weiSoftmax = device.Dropout(weiSoftmax, l.dropoutProb)

	return device.MatrixMultiply3D(weiSoftmax, valObject, 1)
}

func (l *MSAHead) ForUpdate() []*num.Data {
	return l.forUpdate
}
