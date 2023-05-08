package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewEmbedding(
	featuresCount int,
	alphabetSize int,
	contextSize int,
) *Embedding {
	return &Embedding{
		ValEmbedding: num.NewRandNorm(num.NewDims(featuresCount, alphabetSize)),
		PosEmbedding: num.NewRandNorm(num.NewDims(featuresCount, contextSize)),
	}
}

type Embedding struct {
	ValEmbedding *num.Data
	PosEmbedding *num.Data

	output *num.Data
}

func (l *Embedding) Compile(inputs *num.Data) *num.Data {
	l.output = inputs.GetEmbeddings(l.ValEmbedding, l.PosEmbedding)
	return l.output
}

func (l *Embedding) Forward() {
	l.output.Forward()
}

func (l *Embedding) ForUpdate() num.Nodes {
	return num.Nodes{l.ValEmbedding, l.PosEmbedding}
}
