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
	embeddingFeatures := 8
	hiddenLayerSize := 50

	return model.NewSequential(contextLength, miniBatchSize, layer.Layers{
		// embedding table
		//layer.NewEmbedWithPos(embeddingFeatures, alphabetSize),
		//layer.NewEmbed(embeddingFeatures, alphabetSize),
		layer.NewEmbedSA(embeddingFeatures, alphabetSize),
		layer.NewLNorm(),
		//layer.NewSAHead(embeddingFeatures, contextLength),
		//layer.NewLNorm(),
		// main hidden layer (iSize needs: contextLength * embeddingFeatures)
		//layer.NewLNorm(),

		layer.NewFC(hiddenLayerSize, num.ReLuGain),
		layer.NewBias(),
		layer.NewReLu(),

		// second hidden layer
		//layer.NewFC(30, num.TanhGain),
		//layer.NewBias(),
		//layer.NewTanh(),

		// output layer
		layer.NewFC(alphabetSize, num.LinearGain),
		//layer.NewBias(),
	})
}
