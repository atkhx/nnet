package pkg

import (
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/layer"
	"github.com/atkhx/nnet/model"
	"github.com/atkhx/nnet/num"
	"github.com/atkhx/nnet/optimizer"
)

const (
	ContextLength = 256
	MiniBatchSize = 16

	EmbeddingFeatures = 384
	HeadSize          = 64
	HeadsCount        = 6
	blocksCount       = 6
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

	initWeight := &initializer.InitWeightFixed{NormK: 0.02}

	SABlock := func() layer.Layers {
		return []layer.Layer{
			layer.NewResidual(
				layer.Layers{
					layer.NewLNorm(),
					layer.NewSAMultiHead(embeddingFeatures, headSize, headsCount, initWeight),
					// out: [ headSize, contextLength, batchSize ]
					layer.NewLinear(embeddingFeatures, initWeight),
					// out: [ embeddingFeatures, contextLength, batchSize ]
				},
			),
			layer.NewResidual(
				// the goal of this block https://youtu.be/XowwKOAWYoQ?t=1297
				layer.Layers{
					layer.NewLNorm(),
					// out: [ embeddingFeatures, contextLength, batchSize ]
					layer.NewLinear(4*embeddingFeatures, initWeight),
					layer.NewReLu(),
					layer.NewLinear(embeddingFeatures, initWeight),
					// out: [ embeddingFeatures, contextLength, batchSize ]
				},
			),
		}
	}
	layers := layer.Layers{}

	//---Embedding table------------------------------------------------------
	layers = append(layers,
		layer.NewEmbedding(embeddingFeatures, alphabetSize, contextLength, initWeight),
		// out: [ embeddingFeatures, contextLength, batchSize ]
	)

	//---SA Blocks-----------------------------------------------
	for i := 0; i < blocksCount; i++ {
		layers = append(layers, SABlock()...)
	}
	// out: [ embeddingFeatures, contextLength, batchSize ]

	//---Probabilities--------------------------------------------------------
	layers = append(layers,
		layer.NewLNorm(),
		layer.NewLinear(alphabetSize, initWeight),
		// out: [ alphabetSize, contextLength, batchSize ]
	)

	//---Adopt probs to 2D----------------------------------------------------
	layers = append(layers,
		layer.NewReshape(num.NewDims(alphabetSize, miniBatchSize*contextLength)),
		// out: [ alphabetSize, contextLength * batchSize ]
	)

	//return model.NewSequential(inDims, layers, optimizer.VanilaSGD(0.0003))
	return model.NewSequential(inDims, layers, optimizer.Adadelta(optimizer.Ro, optimizer.Eps))
}
