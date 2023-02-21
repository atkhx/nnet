package pkg

import (
	"github.com/atkhx/nnet/layer/fc"
	"github.com/atkhx/nnet/layer/softmax"
	"github.com/atkhx/nnet/net"
)

func CreateNN(alphabetSize, wordLen int) *net.FeedForward {
	return net.New(alphabetSize, wordLen, 1, net.Layers{
		fc.New(fc.OutputSizes(alphabetSize, 1, 1)),
		//activation.NewSigmoid(),
		//fc.New(fc.OutputSizes(alphabetSize, 1, 1)),
		softmax.New(),
	})
}
