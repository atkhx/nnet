package layer

import (
	"math"

	"github.com/atkhx/nnet/num"
)

func NewSAHead(
	embeddingFeatures int,
	contextLength int,
) *SAHead {
	return &SAHead{
		embeddingFeatures: embeddingFeatures,
		contextLength:     contextLength,

		KeyLinear: NewFC(embeddingFeatures, num.LinearGain),
		QryLinear: NewFC(embeddingFeatures, num.LinearGain),
		ValLinear: NewFC(embeddingFeatures, num.LinearGain),
	}
}

type SAHead struct {
	embeddingFeatures int
	contextLength     int

	batchSize int
	iSize     int

	KeyLinear *FC
	QryLinear *FC
	ValLinear *FC

	keyObj *num.Data
	qryObj *num.Data
	valObj *num.Data
	weiObj *num.Data

	inputsObj *num.Data
	outputObj *num.Data

	k float64
}

func (l *SAHead) Compile(bSize int, inputs *num.Data) *num.Data {
	l.batchSize = bSize
	l.inputsObj = inputs

	inputsLen := len(inputs.GetData())

	l.iSize = inputsLen / bSize

	l.keyObj = l.KeyLinear.Compile(bSize, inputs)
	l.qryObj = l.QryLinear.Compile(bSize, inputs)
	l.valObj = l.ValLinear.Compile(bSize, inputs)

	l.weiObj = num.New(l.batchSize * l.embeddingFeatures * l.embeddingFeatures)
	l.outputObj = num.New(bSize * l.embeddingFeatures * l.embeddingFeatures)

	l.k = math.Pow(float64(l.embeddingFeatures), -0.5)
	return l.outputObj
}

func (l *SAHead) Forward() {
	l.KeyLinear.Forward() // keyObj filled
	l.QryLinear.Forward() // qryObj filled
	l.ValLinear.Forward() // valObj filled

	// keyObj length = 40 = 5 x 8 (5 smb X 8 features)
	// qryObj length = 40 = 5 x 8 (5 smb X 8 features)
	// weiObj length = 64

	l.keyObj.DotTo(l.weiObj, l.qryObj, l.batchSize*l.embeddingFeatures) // weiObj filled
	//l.weiObj.Print(l.batchSize)

	wei := l.weiObj.MulScalar(l.k)
	wei = wei.Tril(l.batchSize, math.Inf(-1))
	//wei.Print(l.batchSize)
	wei = wei.Softmax(l.batchSize * l.embeddingFeatures)
	//wei.Print(l.batchSize)
	//os.Exit(1)

	l.valObj.DotTo(l.outputObj, wei, l.batchSize*l.embeddingFeatures) // outputObj
}

func (l *SAHead) ForUpdate() num.Nodes {
	return num.Nodes{
		l.keyObj,
		l.qryObj,
		l.valObj,
	}
}
