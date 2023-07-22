package pkg

import (
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/layer"
	"github.com/atkhx/nnet/model"
	"github.com/atkhx/nnet/num"
	"github.com/atkhx/nnet/optimizer"
)

const (
	ContextLength = 64
	MiniBatchSize = 10

	HeadSize          = 64
	HeadsCount        = 4
	EmbeddingFeatures = HeadSize * HeadsCount
	blocksCount       = 6
	HeadLinearSize    = 4
)

// blocksCount не так сильно влияют на скорость
// context 256, batch 10, headSize 64, headCount 4
// 16 блоков - 15 секунд
// 1 блок - 2.5 секунды

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

	layers := layer.Layers{}

	//---Embedding table------------------------------------------------------
	layers = append(layers,
		layer.NewEmbedding(embeddingFeatures, alphabetSize, contextLength),
		// out: [ embeddingFeatures, contextLength, batchSize ]
	)

	createSABlock := func() layer.Layers {
		return []layer.Layer{
			layer.NewResidual(
				layer.Layers{
					layer.NewLNorm(),
					layer.NewMSAMultiHead(embeddingFeatures, headSize, headsCount, initWeight),
					// out: [ headSize * headsCount, contextLength, batchSize ]
					layer.NewLinear(embeddingFeatures, initWeight),
					// out: [ embeddingFeatures, contextLength, batchSize ]
				},
			),
			layer.NewResidual(
				// the goal of this block https://youtu.be/XowwKOAWYoQ?t=1297
				layer.Layers{
					layer.NewLNorm(),
					// out: [ embeddingFeatures, contextLength, batchSize ]
					layer.NewLinear(HeadLinearSize*embeddingFeatures, initWeight),
					layer.NewReLu(),
					layer.NewLinear(embeddingFeatures, initWeight),
					// out: [ embeddingFeatures, contextLength, batchSize ]
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

	return model.NewSequential(inDims, layers, optimizer.Adam(0.9, 0.98, 3e-4, 0.000000001))
}
