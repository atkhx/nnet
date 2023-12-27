package layer

import (
	"encoding/json"
	"math"

	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/num"
)

func NewSAMultiHeadRopeCols(
	featuresCount int,
	headSize int,
	headsCount int,
	contextLength int,
	dropoutProb float32,
	initWeights initializer.Initializer,
	provideWeights func(qw, kw, vw *num.Data),
) *SAMultiHeadRopeCols {
	return &SAMultiHeadRopeCols{
		initWeights:    initWeights,
		featuresCount:  featuresCount,
		contextLength:  contextLength,
		headsCount:     headsCount,
		headSize:       headSize,
		dropoutProb:    dropoutProb,
		provideWeights: provideWeights,
	}
}

type SAMultiHeadRopeCols struct {
	QryWeights *num.Data
	KeyWeights *num.Data
	ValWeights *num.Data

	forUpdate []*num.Data

	initWeights    initializer.Initializer
	provideWeights func(qw, kw, vw *num.Data)

	featuresCount int
	contextLength int
	headsCount    int
	headSize      int
	dropoutProb   float32
}

func (l *SAMultiHeadRopeCols) Compile(device nnet.Device, inputs *num.Data) *num.Data {
	weightK := l.initWeights.GetNormK(device.GetDataLength(inputs))
	batchSize := inputs.Dims.D

	l.QryWeights = device.NewDataRandNormWeighted(num.NewDims(l.featuresCount, l.featuresCount), weightK)
	l.KeyWeights = device.NewDataRandNormWeighted(num.NewDims(l.featuresCount, l.featuresCount), weightK)
	l.ValWeights = device.NewDataRandNormWeighted(num.NewDims(l.featuresCount, l.featuresCount), weightK)

	l.forUpdate = []*num.Data{l.QryWeights, l.KeyWeights, l.ValWeights}

	bx := device.Transpose(inputs) // bx - vertical

	// Extract qkv-objects
	qryObject := device.MatrixMultiply(l.QryWeights, bx, 1)
	keyObject := device.MatrixMultiply(l.KeyWeights, bx, 1)
	valObject := device.MatrixMultiply(l.ValWeights, bx, 1)

	// Apply RoPE
	qryObject = device.RopeCols(qryObject, l.featuresCount, l.headSize, l.contextLength)
	keyObject = device.RopeCols(keyObject, l.featuresCount, l.headSize, l.contextLength)

	// Reshape qkv-objects
	reshapeToDims := num.NewDims(l.contextLength, l.headSize, l.headsCount*batchSize)

	qryObject = device.Reshape(qryObject, reshapeToDims)
	keyObject = device.Reshape(keyObject, reshapeToDims)
	valObject = device.Reshape(valObject, reshapeToDims)

	// Transpose q and v
	qryObject = device.Transpose(qryObject)
	valObject = device.Transpose(valObject)

	// Extract weiObject
	k := float32(math.Pow(float64(l.headSize), -0.5))
	weiObject := device.MatrixMultiply(qryObject, keyObject, k)

	// Apply triangle lower softmax
	weiSoftmax := device.TriangleLowerSoftmax(weiObject)

	// Get MHA-output objects
	bx = device.MatrixMultiply(weiSoftmax, valObject, 1) // bx - horizontal stacked

	// Transpose output before reshape
	bx = device.Transpose(bx) // bx - vertical stacked

	// Reshape output back to big matrix (instead of concatenation)
	bx = device.Reshape(bx, num.NewDims(l.contextLength, l.featuresCount, batchSize)) // bx - vertical

	out := device.Transpose(bx) // bx - horizontal

	return out
}

func (l *SAMultiHeadRopeCols) ForUpdate() []*num.Data {
	return l.forUpdate
}

type SAMultiHeadRope2Config struct {
	QryWeights []float32
	KeyWeights []float32
	ValWeights []float32
}

func (l *SAMultiHeadRopeCols) MarshalJSON() ([]byte, error) {
	config := SAMultiHeadRope2Config{
		QryWeights: l.QryWeights.Data.GetData(),
		KeyWeights: l.KeyWeights.Data.GetData(),
		ValWeights: l.ValWeights.Data.GetData(),
	}
	return json.Marshal(config)
}

func (l *SAMultiHeadRopeCols) UnmarshalJSON(bytes []byte) error {
	config := SAMultiHeadRope2Config{
		QryWeights: l.QryWeights.Data.GetData(),
		KeyWeights: l.KeyWeights.Data.GetData(),
		ValWeights: l.ValWeights.Data.GetData(),
	}
	return json.Unmarshal(bytes, &config)
}

func (l *SAMultiHeadRopeCols) LoadFromProvider() {
	l.provideWeights(l.QryWeights, l.KeyWeights, l.ValWeights)
}
