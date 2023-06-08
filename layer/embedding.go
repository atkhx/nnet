package layer

import (
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/num"
)

func NewEmbedding(
	featuresCount int,
	alphabetSize int,
	contextSize int,
	initWeights initializer.Initializer,
) *Embedding {
	return &Embedding{
		ValEmbedding: num.NewRandNorm(num.NewDims(featuresCount, alphabetSize)),
		PosEmbedding: num.NewRandNorm(num.NewDims(featuresCount, contextSize)),
		initWeights:  initWeights,
	}
}

type Embedding struct {
	initWeights initializer.Initializer

	ValEmbedding *num.Data
	PosEmbedding *num.Data

	inputsObj *num.Data
	outputObj *num.Data
	forUpdate num.Nodes
}

func (l *Embedding) Compile(inputs *num.Data) *num.Data {
	normK := l.initWeights.GetNormK(len(inputs.Data))

	l.ValEmbedding.MulScalar(normK)
	l.PosEmbedding.MulScalar(normK)

	l.inputsObj = inputs
	l.outputObj = inputs.GetEmbeddings(l.ValEmbedding, l.PosEmbedding)
	l.forUpdate = num.Nodes{l.ValEmbedding, l.PosEmbedding}

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
