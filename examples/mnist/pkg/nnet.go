package pkg

import (
	"github.com/atkhx/nnet/dataset/mnist"
	"github.com/atkhx/nnet/layer/activation"
	"github.com/atkhx/nnet/layer/conv"
	"github.com/atkhx/nnet/layer/fc"
	"github.com/atkhx/nnet/layer/maxpooling"
	"github.com/atkhx/nnet/layer/softmax"
	"github.com/atkhx/nnet/net"
)

func CreateConvNet() *net.FeedForward {
	return net.New(
		mnist.ImageWidth,
		mnist.ImageHeight,
		mnist.ImageDepth,
		net.Layers{
			conv.New(
				conv.FilterSize(3),
				conv.FiltersCount(10),
				conv.Padding(1),
			),
			activation.NewReLu(),
			maxpooling.New(
				maxpooling.FilterSize(2),
				maxpooling.Stride(2),
			),

			conv.New(
				conv.FilterSize(3),
				conv.FiltersCount(10),
			),
			activation.NewReLu(),
			maxpooling.New(
				maxpooling.FilterSize(2),
				maxpooling.Stride(2),
			),

			fc.New(fc.OutputSizes(10, 1, 1)),
			softmax.New(),
		},
	)
}
