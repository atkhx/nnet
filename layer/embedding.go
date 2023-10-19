package layer

import (
	"github.com/atkhx/nnet"
)

func NewEmbedding[data any](
	valEmbedding data,
	posEmbedding data,
) *Embedding[data] {
	return &Embedding[data]{
		ValEmbedding: valEmbedding,
		posEmbedding: posEmbedding,
		forUpdate:    []data{valEmbedding},
	}
}

type Embedding[data any] struct {
	ValEmbedding data
	posEmbedding data

	forUpdate []data
}

func (l *Embedding[data]) Compile(device nnet.Device[data], inputs data) data {
	return device.Embeddings(inputs, l.ValEmbedding, l.posEmbedding)
}

func (l *Embedding[data]) ForUpdate() []data {
	return l.forUpdate
}
