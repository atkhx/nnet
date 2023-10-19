package model

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/layer"
	"github.com/atkhx/nnet/num"
)

func NewTransformer[data any](
	contextLength,
	embeddingFeatures,
	headsCount,
	headSize,
	headLinearSize,
	blocksCount,
	alphabetSize,
	miniBatchSize int,
	dropout float32,
	device nnet.Device[data],
	modelOptimizer Optimizer[data],
) *Sequential[data] {
	inDims := num.Dims{
		W: contextLength,
		H: miniBatchSize,
		D: 1,
	}

	//initWeight := &initializer.InitWeightFixed{NormK: 0.02}
	//initWeight := &initializer.InitWeightFixed{NormK: 0.010}
	initWeight := &initializer.InitWeightFixed{NormK: 0.007}

	layers := layer.Layers[data]{}

	//---Embedding table------------------------------------------------------
	layers = append(layers,
		layer.NewEmbedding(
			device.NewTokenEmbeddingTable(embeddingFeatures, alphabetSize),
			device.NewPositionEmbeddingTable(embeddingFeatures, contextLength),
		),
		// out: [ embeddingFeatures, contextLength, batchSize ]
	)

	createSABlock := func() layer.Layers[data] {
		return []nnet.Layer[data]{
			layer.NewResidual[data](
				layer.Layers[data]{
					layer.NewLNorm[data](),
					layer.NewMSAMultiHead[data](embeddingFeatures, headSize, headsCount, dropout, initWeight),
					// out: [ headSize * headsCount, contextLength, batchSize ]
					layer.NewLinear[data](embeddingFeatures, initWeight),
					// out: [ embeddingFeatures, contextLength, batchSize ]
					layer.NewDropout[data](dropout),
				},
			),
			layer.NewResidual[data](
				// the goal of this block https://youtu.be/XowwKOAWYoQ?t=1297
				layer.Layers[data]{
					layer.NewLNorm[data](),
					// out: [ embeddingFeatures, contextLength, batchSize ]
					layer.NewLinear[data](headLinearSize*embeddingFeatures, initWeight),
					layer.NewReLu[data](),
					layer.NewLinear[data](embeddingFeatures, initWeight),
					// out: [ embeddingFeatures, contextLength, batchSize ]
					layer.NewDropout[data](dropout),
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
		layer.NewLNorm[data](),
		layer.NewLinear[data](alphabetSize, initWeight),
		// out: [ alphabetSize, contextLength, batchSize ]
	)

	//---Adopt probs to 2D----------------------------------------------------
	layers = append(layers,
		layer.NewReshape[data](num.NewDims(alphabetSize, miniBatchSize*contextLength)),
		// out: [ alphabetSize, contextLength * batchSize ]
	)

	return NewSequential(inDims, layers, device, modelOptimizer)
}
