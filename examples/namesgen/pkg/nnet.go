package pkg

import (
	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/layer/activation"
	"github.com/atkhx/nnet/layer/fc"
	"github.com/atkhx/nnet/layer/reshape"
	"github.com/atkhx/nnet/net"
)

func CreateNN(
	alphabetSize int,
	contextSize int,
	miniBatchSize int,
) *net.FeedForward {
	embeddingFeatures := 10
	hiddenLayerSize := 50
	return net.New(net.Layers{
		// embedding table
		fc.New(
			fc.WithInputSize(alphabetSize),
			fc.WithLayerSize(embeddingFeatures),
			fc.WithGain(data.LinearGain),
			fc.WithBatchSize(miniBatchSize),
		),

		// reshape data to embeddingFeatures x wordLength
		reshape.New(func(iw, ih, id int) (int, int, int) {
			return iw * contextSize, ih / contextSize, id
		}),

		// main hidden layer
		fc.New(
			fc.WithInputSize(contextSize*embeddingFeatures),
			fc.WithLayerSize(hiddenLayerSize),
			fc.WithGain(data.TanhGain),
			fc.WithBatchSize(miniBatchSize),
			fc.WithBiases(true),
		),

		activation.NewTanh(),

		fc.New(
			fc.WithInputSize(hiddenLayerSize),
			fc.WithLayerSize(alphabetSize),
			fc.WithBiases(true),
			fc.WithGain(data.LinearGain),
			fc.WithBatchSize(miniBatchSize),
		),
	})
}
