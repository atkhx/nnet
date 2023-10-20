package model

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/layer"
	"github.com/atkhx/nnet/num"
)

func NewTransformer(
	contextLength,
	embeddingFeatures,
	headsCount,
	headSize,
	headLinearSize,
	blocksCount,
	alphabetSize,
	miniBatchSize int,
	dropout float32,
	device nnet.Device,
	modelOptimizer Optimizer,
) *Sequential {
	inDims := num.Dims{
		W: contextLength,
		H: miniBatchSize,
		D: 1,
	}

	//initWeight := &initializer.InitWeightFixed{NormK: 0.02}
	//initWeight := &initializer.InitWeightFixed{NormK: 0.010}
	initWeight := &initializer.InitWeightFixed{NormK: 0.007}

	layers := layer.Layers{}

	//---Embedding table------------------------------------------------------
	layers = append(layers,
		layer.NewEmbedding(
			device.NewTokenEmbeddingTable(embeddingFeatures, alphabetSize),
			device.NewPositionEmbeddingTable(embeddingFeatures, contextLength),
		),
		// out: [ embeddingFeatures, contextLength, batchSize ]
	)

	createSABlock := func() layer.Layers {
		return []nnet.Layer{
			layer.NewResidual(
				layer.Layers{
					layer.NewLNorm(),
					layer.NewMSAMultiHead(embeddingFeatures, headSize, headsCount, dropout, initWeight),
					// out: [ headSize * headsCount, contextLength, batchSize ]
					layer.NewLinear(embeddingFeatures, initWeight),
					// out: [ embeddingFeatures, contextLength, batchSize ]
					layer.NewDropout(dropout),
				},
			),
			layer.NewResidual(
				// the goal of this block https://youtu.be/XowwKOAWYoQ?t=1297
				layer.Layers{
					layer.NewLNorm(),
					// out: [ embeddingFeatures, contextLength, batchSize ]
					layer.NewLinear(headLinearSize*embeddingFeatures, initWeight),
					layer.NewReLu(),
					layer.NewLinear(embeddingFeatures, initWeight),
					// out: [ embeddingFeatures, contextLength, batchSize ]
					layer.NewDropout(dropout),
				},
			),
		}
	}

	//---SA Blocks-----------------------------------------------
	for i := 0; i < blocksCount; i++ {
		layers = append(layers, createSABlock()...)
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

	return NewSequential(inDims, layers, device, modelOptimizer)
}
