package model

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/layer"
	"github.com/atkhx/nnet/num"
)

func NewLLaMaExperiment(
	contextLength,
	embeddingFeatures,
	headsCount,
	kvHeadsCount,
	headSize,
	headLinearSize,
	blocksCount,
	alphabetSize,
	miniBatchSize int,
	dropout float32,
	initWeightK float32,
	device nnet.Device,
	modelOptimizer Optimizer,
) *Sequential {
	inDims := num.Dims{
		W: contextLength,
		H: miniBatchSize,
		D: 1,
	}

	initWeightK = 0.007

	initWeight := &initializer.InitWeightFixed{NormK: initWeightK}
	initWeightRMSMul := &initializer.InitWeightFixed{NormK: 3.7}
	//initWeightRMSMul := &initializer.InitWeightFixed{NormK: 0.7}
	//initWeightRMSMul := &initializer.InitWeightFixed{NormK: initWeightK}

	layers := layer.Layers{}

	embeddings := device.NewTokenEmbeddingTable(embeddingFeatures, alphabetSize)
	for i := range embeddings.Data.GetData() {
		//embeddings.Data.GetData()[i] *= 0.02
		embeddings.Data.GetData()[i] *= initWeightK
	}

	//---Embedding table------------------------------------------------------
	layers = append(layers,
		layer.NewEmbeddings(embeddings, nil),
		// out: [ embeddingFeatures, contextLength, batchSize ]
	)

	createSABlock := func() layer.Layers {
		return []nnet.Layer{
			layer.NewResidual(
				layer.Layers{
					layer.NewRMSLNorm(),
					layer.NewMulRows(embeddingFeatures, initWeightRMSMul, nil),
					// out: [ embeddingFeatures, contextLength, batchSize ]

					//layer.NewSAMultiHeadRope(embeddingFeatures, headSize, headsCount, contextLength, dropout, initWeight, nil),
					layer.NewSAMultiHeadRopeCols(embeddingFeatures, headSize, headsCount, contextLength, dropout, initWeight, nil),
					// out: [ headSize * headsCount, contextLength, batchSize ]
					layer.NewLinear(embeddingFeatures, initWeight, false, nil),
					// out: [ embeddingFeatures, contextLength, batchSize ]
					layer.NewDropout(dropout),
				},
			),
			layer.NewResidual(
				layer.Layers{
					// layer.NewLNorm(),
					layer.NewRMSLNorm(),
					layer.NewMulRows(embeddingFeatures, initWeightRMSMul, nil),
					// out: [ embeddingFeatures, contextLength, batchSize ]

					layer.NewSwiGLU(embeddingFeatures, headLinearSize, initWeight, nil),
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
	//---Probabilities--------------------------------------------------------
	layers = append(layers,
		layer.NewRMSLNorm(),
		layer.NewMulRows(embeddingFeatures, initWeightRMSMul, nil),
		//layer.NewLinear(alphabetSize, initWeight, false, nil),
		layer.NewLinearWithWeights(device.Transpose(embeddings)),
		// out: [ alphabetSize, contextLength, batchSize ]
	)

	//---Adopt probs to 2D----------------------------------------------------
	layers = append(layers,
		layer.NewReshape(num.NewDims(alphabetSize, miniBatchSize*contextLength)),
		// out: [ alphabetSize, contextLength * batchSize ]
	)
	return NewSequential(inDims, layers, device, modelOptimizer)
}
