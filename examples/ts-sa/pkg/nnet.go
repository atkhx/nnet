package pkg

import (
	"github.com/atkhx/nnet/layer"
	"github.com/atkhx/nnet/model"
	"github.com/atkhx/nnet/num"
)

const (
	ContextLength = 64
	MiniBatchSize = 16

	EmbeddingFeatures = 32
	HeadSize          = 8
	HeadsCount        = 4
)

func CreateNN(
	alphabetSize int,
	miniBatchSize int,
) *model.Sequential {
	contextLength := ContextLength
	embeddingFeatures := EmbeddingFeatures
	headSize := HeadSize
	headsCount := HeadsCount

	inDims := num.Dims{
		W: contextLength,
		H: miniBatchSize,
		D: 1,
	}

	return model.NewSequential(inDims, layer.Layers{
		// embedding table
		layer.NewEmbedding(embeddingFeatures, alphabetSize, contextLength),
		// out: [ embeddingFeatures, contextLength, batchSize ]

		//------------------------------------------------------------------------
		// Block 1
		// SA-MultiHead
		layer.NewSAMultiHead(embeddingFeatures, headSize, headsCount),
		layer.NewFC(num.NewDims(embeddingFeatures, headsCount*headSize), num.LinearGain),
		layer.NewBias(num.NewDims(embeddingFeatures, contextLength)),
		// out: [ embeddingFeatures, contextLength, batchSize ]

		// Non-linearity in block
		layer.NewFC(num.NewDims(4*embeddingFeatures, embeddingFeatures), num.ReLuGain),
		layer.NewBias(num.NewDims(4*embeddingFeatures, contextLength)),
		layer.NewReLu(),
		layer.NewFC(num.NewDims(embeddingFeatures, 4*embeddingFeatures), num.LinearGain),
		layer.NewBias(num.NewDims(embeddingFeatures, contextLength)),
		//out: [ embeddingFeatures, contextLength, batchSize ]

		//------------------------------------------------------------------------
		// Block 2
		// SA-MultiHead
		layer.NewSAMultiHead(embeddingFeatures, headSize, headsCount),
		layer.NewFC(num.NewDims(embeddingFeatures, headsCount*headSize), num.LinearGain),
		layer.NewBias(num.NewDims(embeddingFeatures, contextLength)),
		// out: [ embeddingFeatures, contextLength, batchSize ]

		// Non-linearity in block
		layer.NewFC(num.NewDims(4*embeddingFeatures, embeddingFeatures), num.ReLuGain),
		layer.NewBias(num.NewDims(4*embeddingFeatures, contextLength)),
		layer.NewReLu(),
		layer.NewFC(num.NewDims(embeddingFeatures, 4*embeddingFeatures), num.LinearGain),
		layer.NewBias(num.NewDims(embeddingFeatures, contextLength)),
		//out: [ embeddingFeatures, contextLength, batchSize ]

		//------------------------------------------------------------------------
		// Probabilities
		layer.NewFC(num.NewDims(alphabetSize, embeddingFeatures), num.LinearGain),
		layer.NewBias(num.NewDims(alphabetSize, contextLength)),
		// out: [ alphabetSize, contextLength, batchSize ]

		// Adopt probs to 2D
		layer.NewReshape(num.NewDims(alphabetSize, miniBatchSize*contextLength)),
		// out: [ alphabetSize, contextLength * batchSize ]
		// each row is probabilities for the next symbol
	})
}
