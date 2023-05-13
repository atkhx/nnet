package pkg

import (
	"github.com/atkhx/nnet/layer"
	"github.com/atkhx/nnet/model"
	"github.com/atkhx/nnet/num"
)

func CreateNN(
	alphabetSize int,
	contextLength int,
	miniBatchSize int,
) *model.Sequential {
	embeddingFeatures := 8
	headSize := 16

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
		layer.NewSAHead(embeddingFeatures, headSize),

		//layer.NewSAMultiHead(embeddingFeatures, headSize, 4),
		layer.NewFC(num.NewDims(embeddingFeatures, 1*headSize), num.LinearGain),
		layer.NewBias(num.NewDims(embeddingFeatures, contextLength)),
		// out: [ embeddingFeatures, contextLength, batchSize ]

		// Non-linearity in block-1
		//layer.NewFC(num.NewDims(4*embeddingFeatures, embeddingFeatures), num.ReLuGain),
		//layer.NewBias(num.NewDims(4*embeddingFeatures, contextLength)),
		//layer.NewReLu(),
		//layer.NewFC(num.NewDims(embeddingFeatures, 4*embeddingFeatures), num.LinearGain),
		//layer.NewBias(num.NewDims(embeddingFeatures, contextLength)),
		// out: [ embeddingFeatures, contextLength, batchSize ]

		//------------------------------------------------------------------------
		// Block 2
		// SA-MultiHead
		//layer.NewSAMultiHead(embeddingFeatures, headSize, 4),
		//layer.NewFC(num.NewDims(embeddingFeatures, 4*headSize), num.LinearGain),
		//layer.NewBias(num.NewDims(embeddingFeatures, contextLength)),
		//// out: [ embeddingFeatures, contextLength, batchSize ]
		//
		//// Non-linearity in block-2
		//layer.NewFC(num.NewDims(4*embeddingFeatures, embeddingFeatures), num.ReLuGain),
		//layer.NewBias(num.NewDims(4*embeddingFeatures, contextLength)),
		//layer.NewReLu(),
		//layer.NewFC(num.NewDims(embeddingFeatures, 4*embeddingFeatures), num.LinearGain),
		//layer.NewBias(num.NewDims(embeddingFeatures, contextLength)),
		// out: [ embeddingFeatures, contextLength, batchSize ]

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
