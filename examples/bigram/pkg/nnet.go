package pkg

import (
	"github.com/atkhx/nnet/layer/fc"
	"github.com/atkhx/nnet/layer/softmax"
	"github.com/atkhx/nnet/net"
)

func CreateNN(alphabetSize, wordLen int) *net.FeedForward {
	// input - one-hot vectors presented word (batchSize = wordLen)
	return net.New(net.Layers{
		fc.New(
			fc.WithInputSize(alphabetSize),
			fc.WithLayerSize(alphabetSize),
		),
		softmax.New(),
	})
}
