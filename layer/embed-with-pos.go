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

	outputObj *num.Data
	inputsObj *num.Data

	WeightsVal num.Float64s // (storable)
	WeightsPos num.Float64s // (storable)
}

func (l *EmbedWithPos) Compile(bSize int, inputs *num.Data) *num.Data {
	inputsLen := len(inputs.GetData())
	l.iSize = inputsLen / bSize
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
		l.inputIdxByVal = make([]int, inputsLen)
		l.inputIdxByPos = make([]int, inputsLen)
	}

	{ // buffers to store intermediate results
		l.tokensByValBuffer = num.New(outputSize)
		l.tokensByPosBuffer = num.New(outputSize)
	}

	l.inputsObj = inputs
	l.outputObj = num.New(outputSize)
	return l.outputObj
}

func (l *EmbedWithPos) Forward() {
	inputs := l.inputsObj.GetData()
	for i, pair := range num.GetRepeatedPosPairs(len(inputs), l.iSize) {
		l.inputIdxByVal[i] = int(inputs[pair[0]])
		l.inputIdxByPos[i] = pair[1]
	}

	l.embeddedByValObj.GetEmbeddedTo(l.tokensByValBuffer, l.featuresCount, l.inputIdxByVal)
	l.embeddedByPosObj.GetEmbeddedTo(l.tokensByPosBuffer, l.featuresCount, l.inputIdxByPos)

	l.tokensByValBuffer.AddTo(l.outputObj, l.tokensByPosBuffer)
}

func (l *EmbedWithPos) ForUpdate() num.Nodes {
	return num.Nodes{
		l.embeddedByValObj,
		l.embeddedByPosObj,
	}
}
