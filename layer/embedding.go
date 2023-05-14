package layer

import (
	"fmt"

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

	outputObj *num.Data
	forUpdate num.Nodes
}

func (l *Embedding) Compile(inputs *num.Data) *num.Data {
	l.outputObj = inputs.GetEmbeddings(l.ValEmbedding, l.PosEmbedding)
	l.forUpdate = num.Nodes{l.ValEmbedding, l.PosEmbedding}

	fmt.Println("Emb\t", l.ValEmbedding.Dims, l.PosEmbedding.Dims, "out", l.outputObj.Dims)
	return l.outputObj
}

func (l *Embedding) Forward() {
	l.outputObj.Forward()
}

func (l *Embedding) Backward() {
	l.outputObj.Backward()
}

func (l *Embedding) ForUpdate() num.Nodes {
	return l.forUpdate
}
