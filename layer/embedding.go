package layer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/num"
)

func NewEmbedding(
	valEmbedding *num.Data,
	posEmbedding *num.Data,
) *Embedding {
	return &Embedding{
		ValEmbedding: valEmbedding,
		posEmbedding: posEmbedding,
		forUpdate:    []*num.Data{valEmbedding},
	}
}

func NewEmbeddingWithSkipTraining(
	valEmbedding *num.Data,
	posEmbedding *num.Data,
) *Embedding {
	return &Embedding{
		ValEmbedding: valEmbedding,
		posEmbedding: posEmbedding,
		skipTraining: true,
		forUpdate:    []*num.Data{},
	}
}

type Embedding struct {
	ValEmbedding *num.Data
	posEmbedding *num.Data
	skipTraining bool

	forUpdate []*num.Data
}

func (l *Embedding) Compile(device nnet.Device, inputs *num.Data) *num.Data {
	return device.Embeddings(inputs, l.ValEmbedding, l.posEmbedding)
}

func (l *Embedding) ForUpdate() []*num.Data {
	return l.forUpdate
}
