package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewEmbedWithPos(
	featuresCount int,
	alphabetSize int,
) *EmbedWithPos {
	return &EmbedWithPos{
		alphabetSize:  alphabetSize,
		featuresCount: featuresCount,
	}
}

type EmbedWithPos struct {
	featuresCount int
	alphabetSize  int

	iSize int
	bSize int

	inputIdxByVal []int
	inputIdxByPos []int

	embeddedByValObj *num.Data // embedding table for get tokens by symbol code
	embeddedByPosObj *num.Data // embedding table for get tokens by symbol position

	tokensByValBuffer *num.Data // buffer for tokens by symbol code
	tokensByPosBuffer *num.Data // buffer for tokens by symbol position
	tokensSumBuffer   *num.Data // buffer for sum tokens (val + pos); result object (layer output)

	// internal buffers
	WeightsVal num.Float64s // (storable)
	WeightsPos num.Float64s // (storable)

	// buffers from the previous layer
	inputs num.Float64s
}

func (l *EmbedWithPos) Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	l.iSize = len(inputs) / bSize
	l.bSize = bSize

	outputSize := l.featuresCount * l.bSize * l.iSize

	{ // code embedding table initialization
		codeEmbeddingSize := l.featuresCount * l.alphabetSize

		l.WeightsVal = num.NewFloat64sRandNorm(codeEmbeddingSize)
		l.embeddedByValObj = num.Wrap(l.WeightsVal, make(num.Float64s, codeEmbeddingSize))
	}

	{ // position embedding table initialization
		posEmbeddingSize := l.featuresCount * l.iSize
		l.WeightsPos = num.NewFloat64sRandNorm(posEmbeddingSize)
		l.embeddedByPosObj = num.Wrap(l.WeightsPos, make(num.Float64s, posEmbeddingSize))
	}

	{ // buffers to store converted inputs information
		l.inputs = inputs // we have to store it because we need direct data access
		l.inputIdxByVal = make([]int, len(inputs))
		l.inputIdxByPos = make([]int, len(inputs))
	}

	{ // buffers to store intermediate results
		l.tokensByValBuffer = num.New(outputSize)
		l.tokensByPosBuffer = num.New(outputSize)
	}

	// candidate to clever output object
	output := make(num.Float64s, outputSize)
	oGrads := make(num.Float64s, outputSize)
	l.tokensSumBuffer = num.Wrap(output, oGrads)

	return output, oGrads
}

func (l *EmbedWithPos) Forward() {
	for i, pair := range num.GetRepeatedPosPairs(len(l.inputs), l.iSize) {
		l.inputIdxByVal[i] = int(l.inputs[pair[0]])
		l.inputIdxByPos[i] = pair[1]
	}

	l.embeddedByValObj.GetEmbeddedTo(l.tokensByValBuffer, l.featuresCount, l.inputIdxByVal)
	l.embeddedByPosObj.GetEmbeddedTo(l.tokensByPosBuffer, l.featuresCount, l.inputIdxByPos)

	l.tokensByValBuffer.AddTo(l.tokensSumBuffer, l.tokensByPosBuffer)
}

func (l *EmbedWithPos) Backward() {
	l.tokensSumBuffer.CalcGrad()
}

func (l *EmbedWithPos) ResetGrads() {
	l.tokensSumBuffer.ResetGrad()
}

func (l *EmbedWithPos) ForUpdate() num.Nodes {
	return num.Nodes{
		l.embeddedByValObj,
		l.embeddedByPosObj,
	}
}
