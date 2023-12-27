package layer

import (
	"encoding/json"
	"math"

	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/num"
)

func NewSAMultiHeadRope(
	featuresCount int,
	headSize int,
	headsCount int,
	contextLength int,
	dropoutProb float32,
	initWeights initializer.Initializer,
	provideWeights func(qw, kw, vw []*num.Data),
) *SAMultiHeadRope {
	return &SAMultiHeadRope{
		initWeights:    initWeights,
		featuresCount:  featuresCount,
		contextLength:  contextLength,
		headsCount:     headsCount,
		headSize:       headSize,
		dropoutProb:    dropoutProb,
		provideWeights: provideWeights,
	}
}

type SAMultiHeadRope struct {
	QryWeights []*num.Data
	KeyWeights []*num.Data
	ValWeights []*num.Data

	forUpdate []*num.Data

	initWeights    initializer.Initializer
	provideWeights func(qw, kw, vw []*num.Data)

	featuresCount int
	contextLength int
	headsCount    int
	headSize      int
	dropoutProb   float32
}

func (l *SAMultiHeadRope) Compile(device nnet.Device, inputs *num.Data) *num.Data {
	weightK := l.initWeights.GetNormK(device.GetDataLength(inputs))

	l.QryWeights = make([]*num.Data, 0, l.headsCount)
	l.KeyWeights = make([]*num.Data, 0, l.headsCount)
	l.ValWeights = make([]*num.Data, 0, l.headsCount)

	for i := 0; i < l.headsCount; i++ {
		l.QryWeights = append(l.QryWeights, device.NewDataRandNormWeighted(num.NewDims(l.headSize, l.featuresCount), weightK))
		l.KeyWeights = append(l.KeyWeights, device.NewDataRandNormWeighted(num.NewDims(l.headSize, l.featuresCount), weightK))
		l.ValWeights = append(l.ValWeights, device.NewDataRandNormWeighted(num.NewDims(l.headSize, l.featuresCount), weightK))
	}

	l.forUpdate = append(l.forUpdate, l.QryWeights...)
	l.forUpdate = append(l.forUpdate, l.KeyWeights...)
	l.forUpdate = append(l.forUpdate, l.ValWeights...)

	var outObjects []*num.Data
	for i := 0; i < l.headsCount; i++ {
		// Extract qkv-objects
		qryObject := device.MatrixMultiply(inputs, l.QryWeights[i], 1)
		keyObject := device.MatrixMultiply(inputs, l.KeyWeights[i], 1)
		valObject := device.MatrixMultiply(inputs, l.ValWeights[i], 1)

		// Apply RoPE
		qryObject = device.Rope(qryObject, i, l.headSize, l.contextLength)
		keyObject = device.Rope(keyObject, i, l.headSize, l.contextLength)

		keyObject = device.Transpose(keyObject)
		weiObject := device.MatrixMultiply(qryObject, keyObject, float32(math.Pow(float64(l.headSize), -0.5)))

		weiSoftmax := device.TriangleLowerSoftmax(weiObject)
		weiSoftmax = device.Dropout(weiSoftmax, l.dropoutProb)

		outObject := device.MatrixMultiply(weiSoftmax, valObject, 1)
		outObjects = append(outObjects, outObject)
	}

	return device.ConcatByRows(outObjects...)
}

func (l *SAMultiHeadRope) ForUpdate() []*num.Data {
	return l.forUpdate
}

type SAMultiHeadRopeConfig struct {
	QryWeights [][]float32
	KeyWeights [][]float32
	ValWeights [][]float32
}

func (l *SAMultiHeadRope) MarshalJSON() ([]byte, error) {
	config := SAMultiHeadRopeConfig{
		QryWeights: make([][]float32, len(l.QryWeights)),
		KeyWeights: make([][]float32, len(l.KeyWeights)),
		ValWeights: make([][]float32, len(l.ValWeights)),
	}

	for i, data := range l.QryWeights {
		config.QryWeights[i] = data.Data.GetData()
	}

	for i, data := range l.KeyWeights {
		config.KeyWeights[i] = data.Data.GetData()
	}

	for i, data := range l.ValWeights {
		config.ValWeights[i] = data.Data.GetData()
	}

	return json.Marshal(config)
}

func (l *SAMultiHeadRope) UnmarshalJSON(bytes []byte) error {
	config := SAMultiHeadRopeConfig{
		QryWeights: make([][]float32, len(l.QryWeights)),
		KeyWeights: make([][]float32, len(l.KeyWeights)),
		ValWeights: make([][]float32, len(l.ValWeights)),
	}

	for i, data := range l.QryWeights {
		config.QryWeights[i] = data.Data.GetData()
	}

	for i, data := range l.KeyWeights {
		config.KeyWeights[i] = data.Data.GetData()
	}

	for i, data := range l.ValWeights {
		config.ValWeights[i] = data.Data.GetData()
	}

	return json.Unmarshal(bytes, &config)
}

func (l *SAMultiHeadRope) LoadFromProvider() {
	l.provideWeights(l.QryWeights, l.KeyWeights, l.ValWeights)
}
