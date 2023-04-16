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
	embeddingFeatures := 2
	hiddenLayerSize := 50

	return model.NewSequential(alphabetSize*contextLength, miniBatchSize, layer.Layers{
		// embedding table
		layer.NewEmbed(embeddingFeatures, contextLength, num.LinearGain),

		// main hidden layer (iSize needs: contextLength * embeddingFeatures)
		layer.NewFC(hiddenLayerSize, num.SigmoidGain),
		layer.NewBias(),
		layer.NewSigmoid(),

		// output layer
		layer.NewFC(alphabetSize, num.LinearGain),
		layer.NewBias(),
	})
}
