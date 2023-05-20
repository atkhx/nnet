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

	//initWeight := &layer.InitWeightFixed{0.005}
	initWeight := &layer.InitWeightFixed{0.02}

	return model.NewSequential(inDims, layer.Layers{
		//------------------------------------------------------------------------
		//---Embedding table------------------------------------------------------
		layer.NewEmbedding(embeddingFeatures, alphabetSize, contextLength, initWeight),
		// out: [ embeddingFeatures, contextLength, batchSize ]
		//layer.NewDebug(10000),
		//---Block 1---SA-MultiHead-----------------------------------------------
		// layer.NewDebug(),
		layer.NewResidual(
			layer.Layers{
				layer.NewLNorm(),
				layer.NewSAMultiHead(embeddingFeatures, headSize, headsCount, initWeight),
				layer.NewFC(num.NewDims(embeddingFeatures, headsCount*headSize), initWeight),
				layer.NewBias(num.NewDims(embeddingFeatures, contextLength)),
				// out: [ embeddingFeatures, contextLength, batchSize ]
			},
		),
		layer.NewResidual(
			layer.Layers{
				layer.NewLNorm(),
				// out: [ embeddingFeatures, contextLength, batchSize ]
				layer.NewFC(num.NewDims(4*embeddingFeatures, embeddingFeatures), initWeight),
				layer.NewBias(num.NewDims(4*embeddingFeatures, contextLength)),
				layer.NewReLu(),
				layer.NewFC(num.NewDims(embeddingFeatures, 4*embeddingFeatures), initWeight),
				layer.NewBias(num.NewDims(embeddingFeatures, contextLength)),
				//out: [ embeddingFeatures, contextLength, batchSize ]
			},
		),

		//---Block 2---SA-MultiHead-----------------------------------------------
		// layer.NewDebug(),
		layer.NewResidual(
			layer.Layers{
				layer.NewLNorm(),
				layer.NewSAMultiHead(embeddingFeatures, headSize, headsCount, initWeight),
				layer.NewFC(num.NewDims(embeddingFeatures, headsCount*headSize), initWeight),
				layer.NewBias(num.NewDims(embeddingFeatures, contextLength)),
				// out: [ embeddingFeatures, contextLength, batchSize ]
			},
		),
		layer.NewResidual(
			layer.Layers{
				layer.NewLNorm(),
				// out: [ embeddingFeatures, contextLength, batchSize ]
				layer.NewFC(num.NewDims(4*embeddingFeatures, embeddingFeatures), initWeight),
				layer.NewBias(num.NewDims(4*embeddingFeatures, contextLength)),
				layer.NewReLu(),
				layer.NewFC(num.NewDims(embeddingFeatures, 4*embeddingFeatures), initWeight),
				layer.NewBias(num.NewDims(embeddingFeatures, contextLength)),
				//out: [ embeddingFeatures, contextLength, batchSize ]
			},
		),

		//---Probabilities--------------------------------------------------------
		//layer.NewDebug(8000),
		layer.NewLNorm(),
		layer.NewFC(num.NewDims(alphabetSize, embeddingFeatures), initWeight),
		//layer.NewLNorm(),
		layer.NewBias(num.NewDims(alphabetSize, contextLength)),
		// out: [ alphabetSize, contextLength, batchSize ]

		//---Adopt probs to 2D----------------------------------------------------
		layer.NewReshape(num.NewDims(alphabetSize, miniBatchSize*contextLength)),
		// out: [ alphabetSize, contextLength * batchSize ]
		// each row is probabilities for the next symbol
		//------------------------------------------------------------------------
	})
}
