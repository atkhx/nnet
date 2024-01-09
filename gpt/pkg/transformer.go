package pkg

import (
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/layer"
	"github.com/atkhx/metal/nn/model"
	"github.com/atkhx/metal/nn/proc"
)

func NewLLaMaExperiment(
	contextLength,
	embeddingFeatures,
	headsCount,
	headSize,
	headLinearSize,
	blocksCount,
	alphabetSize,
	miniBatchSize int,
	dropout float32,
	initWeightK float32,
	device *proc.Device,
	modelOptimizer proc.Optimizer,
) *model.Model {
	inDims := mtl.MTLSize{
		W: contextLength,
		H: miniBatchSize,
		D: 1,
	}

	initWeight := &initializer.InitWeightFixed{NormK: initWeightK}
	initWeightRMSMul := &initializer.InitWeightFixed{NormK: 1}

	layers := layer.Layers{}

	embeddings := device.NewTokenEmbeddingTable(embeddingFeatures, alphabetSize, initWeightK)

	//---Embedding table------------------------------------------------------
	layers = append(layers,
		layer.NewEmbeddings(embeddings, nil),
		// out: [ embeddingFeatures, contextLength, batchSize ]
	)

	createSABlock := func() layer.Layers {
		return []layer.Layer{
			layer.NewResidual(
				layer.Layers{
					layer.NewRMSLNorm(),
					layer.NewMulRows(embeddingFeatures, initWeightRMSMul, nil),
					// out: [ embeddingFeatures, contextLength, batchSize ]

					layer.NewSAMultiHead(embeddingFeatures, headSize, headsCount, contextLength, initWeight, nil),
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
		layer.NewLinearWithWeights(device.Transpose(embeddings)),
		// out: [ alphabetSize, contextLength, batchSize ]
	)

	//---Adopt probs to 2D----------------------------------------------------
	layers = append(layers,
		layer.NewReshape(mtl.NewMTLSize(alphabetSize, miniBatchSize*contextLength)),
		// out: [ alphabetSize, contextLength * batchSize ]
	)
	return model.New(inDims, layers, device, modelOptimizer)
}
