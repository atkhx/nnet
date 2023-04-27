package layer

import (
	"math"
	"os"

	"github.com/atkhx/nnet/num"
)

func NewEmbedSA(
	featuresCount int,
	alphabetSize int,
) *EmbedSA {
	return &EmbedSA{
		alphabetSize:  alphabetSize,
		featuresCount: featuresCount,
		ValLinear:     NewFC(featuresCount*featuresCount, num.LinearGain),
	}
}

type EmbedSA struct {
	featuresCount int
	alphabetSize  int

	contextLength int
	batchSize     int

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

	ValLinear *FC
	valObj    *num.Data
}

func (l *EmbedSA) Compile(bSize int, inputs *num.Data) *num.Data {
	inputsLen := len(inputs.GetData())
	l.contextLength = inputsLen / bSize
	l.batchSize = bSize

	outputSize := l.batchSize * l.featuresCount * l.contextLength

	{ // code embedding table initialization
		codeEmbeddingSize := l.featuresCount * l.alphabetSize

		l.WeightsVal = num.NewFloat64sRandNorm(codeEmbeddingSize)
		l.embeddedByValObj = num.Wrap(l.WeightsVal, make(num.Float64s, codeEmbeddingSize))
	}

	{ // position embedding table initialization todo contextLength
		posEmbeddingSize := l.featuresCount * l.contextLength
		l.WeightsPos = num.NewFloat64sRandNorm(posEmbeddingSize)
		l.embeddedByPosObj = num.Wrap(l.WeightsPos, make(num.Float64s, posEmbeddingSize))
	}

	{
		l.valObj = l.ValLinear.Compile(bSize, inputs)
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
	l.outputObj = num.New(l.batchSize * l.featuresCount * l.featuresCount) // todo contextLength
	return l.outputObj
}

func (l *EmbedSA) Forward() {
	inputs := l.inputsObj.GetData()
	for i, pair := range num.GetRepeatedPosPairs(len(inputs), l.contextLength) {
		l.inputIdxByVal[i] = int(inputs[pair[0]])
		l.inputIdxByPos[i] = pair[1]
	}

	l.embeddedByValObj.GetEmbeddedTo(l.tokensByValBuffer, l.featuresCount, l.inputIdxByVal)
	//l.embeddedByPosObj.GetEmbeddedTo(l.tokensByPosBuffer, l.featuresCount, l.inputIdxByPos)

	k := math.Pow(float64(l.featuresCount), -0.5)
	//out := num.New(l.batchSize * l.featuresCount * l.featuresCount)
	out := num.New(l.batchSize * l.contextLength * l.contextLength)
	// todo here we need conteext len instead of features count
	//l.tokensByValBuffer.DotTo(out, l.tokensByPosBuffer, l.batchSize*l.featuresCount)
	l.tokensByValBuffer.MatrixMultiplyTransposedTo(out, l.embeddedByPosObj, l.batchSize*l.featuresCount)

	//l.tokensByPosBuffer.PrintSized(l.batchSize, l.featuresCount, l.featuresCount)
	out.PrintSized(l.batchSize, l.contextLength, l.contextLength) // 30 x 8 x 8
	os.Exit(1)

	out.MulScalar(k)
	out = out.Tril(l.batchSize, math.Inf(-1))
	out = out.Softmax(l.batchSize * l.featuresCount)
	//out.PrintSized(l.batchSize, l.featuresCount, l.featuresCount) // 30 x 8 x 8
	//os.Exit(1)

	l.ValLinear.Forward()                                         // valObj filled
	l.valObj.DotTo(l.outputObj, out, l.batchSize*l.featuresCount) // outputObj

	//l.valObj.PrintSized(l.batchSize, l.featuresCount, l.featuresCount) // 30 x 1, 1
	//l.outputObj.PrintSized(l.batchSize, l.featuresCount, l.featuresCount) // 30 x 8 x 8
	//os.Exit(1)
}

func (l *EmbedSA) ForUpdate() num.Nodes {
	return num.Nodes{
		l.embeddedByValObj,
		l.embeddedByPosObj,
	}
}
