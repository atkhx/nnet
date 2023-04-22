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
	hiddenLayerSize := 100

	return model.NewSequential(contextLength, miniBatchSize, layer.Layers{
		// embedding table
		layer.NewEmbed(embeddingFeatures, alphabetSize, num.LinearGain),

		// main hidden layer (iSize needs: contextLength * embeddingFeatures)
		layer.NewFC(hiddenLayerSize, num.TanhGain),
		layer.NewBias(),
		layer.NewTanh(),

		// second hidden layer
		//layer.NewFC(100, num.TanhGain),
		//layer.NewBias(),
		//layer.NewTanh(),

		// output layer
		layer.NewFC(alphabetSize, num.LinearGain),
		layer.NewBias(),
	})
}
