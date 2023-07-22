package layer

import (
	"math"

	"github.com/atkhx/nnet/num"
)

func NewTokenEmbeddingTable(featuresCount, alphabetSize int) *num.Data {
	return num.NewRandNorm(num.NewDims(featuresCount, alphabetSize))
}

func NewPositionEmbeddingTable(featuresCount, contextSize int) *num.Data {
	// https://youtu.be/XowwKOAWYoQ?t=582
	result := num.New(num.NewDims(featuresCount, contextSize))
	k := 0
	for j := 0; j < contextSize; j++ {
		for i := 0; i < featuresCount; i++ {
			if i%2 == 0 {
				result.Data[k] = math.Sin(float64(k) / math.Pow(10_000, float64(i+1)/float64(featuresCount)))
			} else {
				result.Data[k] = math.Cos(float64(k) / math.Pow(10_000, float64(i+1)/float64(featuresCount)))
			}
			k++
		}
	}
	return result
}

func NewEmbedding(
	featuresCount int,
	alphabetSize int,
	contextSize int,
) *Embedding {
	return &Embedding{
		ValEmbedding: NewTokenEmbeddingTable(featuresCount, alphabetSize),
		posEmbedding: NewPositionEmbeddingTable(featuresCount, contextSize),
	}
}

type Embedding struct {
	ValEmbedding *num.Data
	posEmbedding *num.Data

	inputsObj *num.Data
	outputObj *num.Data
	forUpdate num.Nodes
}

func (l *Embedding) Compile(inputs *num.Data) *num.Data {
	l.inputsObj = inputs
	l.outputObj = inputs.GetEmbeddings(l.ValEmbedding, l.posEmbedding)
	l.forUpdate = num.Nodes{l.ValEmbedding}

	return l.outputObj
}

func (l *Embedding) ForUpdate() num.Nodes {
	return l.forUpdate
}

func (l *Embedding) GetInputs() *num.Data {
	return l.inputsObj
}

func (l *Embedding) GetOutput() *num.Data {
	return l.outputObj
}
