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
	embeddingFeatures := 32
	hiddenLayerSize := 32

	headSize := 32

	inDims := num.Dims{
		W: contextLength,
		H: miniBatchSize,
		D: 1,
	}

	return model.NewSequential(inDims, layer.Layers{
		// embedding table
		layer.NewEmbedding(embeddingFeatures, alphabetSize, contextLength),
		//layer.NewLNorm(),
		// out: [ embeddingFeatures, contextLength, batchSize ]

		// Block 1
		// SA-MultiHead
		layer.NewSAHead(embeddingFeatures, headSize),
		//layer.NewSAMultiHead(embeddingFeatures, headSize),
		// out: [ headSize, contextLength, batchSize ]
		layer.NewFC(num.NewDims(hiddenLayerSize, headSize), num.ReLuGain), // we don't use batchSize, to make weights shared
		layer.NewBias(num.NewDims(hiddenLayerSize, contextLength)),        // we don't use batchSize, to make bias shared
		layer.NewReLu(),
		// out: [ hiddenLayerSize, contextLength, batchSize ]

		// Block 1
		// SA-MultiHead
		//layer.NewSAMultiHead(hiddenLayerSize, headSize),
		//// out: [ headSize, contextLength, batchSize ]
		//layer.NewFC(num.NewDims(hiddenLayerSize, headSize), num.ReLuGain), // we don't use batchSize, to make weights shared
		//layer.NewBias(num.NewDims(hiddenLayerSize, contextLength)),        // we don't use batchSize, to make bias shared
		//layer.NewReLu(),
		// out: [ hiddenLayerSize, contextLength, batchSize ]

		// Linear layer to make a predictions
		layer.NewReshape(num.NewDims(hiddenLayerSize*contextLength, 1, miniBatchSize)),
		//layer.NewLNorm(),
		layer.NewFC(num.NewDims(alphabetSize, hiddenLayerSize*contextLength), num.LinearGain),
		layer.NewBias(num.NewDims(alphabetSize)),
	})
}
