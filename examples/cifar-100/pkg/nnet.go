package pkg

import (
	cifar_100 "github.com/atkhx/nnet/dataset/cifar-100"
	"github.com/atkhx/nnet/layer/activation"
	"github.com/atkhx/nnet/layer/conv"
	"github.com/atkhx/nnet/layer/fc"
	"github.com/atkhx/nnet/layer/maxpooling"
	"github.com/atkhx/nnet/layer/softmax"
	"github.com/atkhx/nnet/net"
)

func CreateConvNet() *net.FeedForward {
	return net.New(cifar_100.ImageWidth, cifar_100.ImageHeight, 3, net.Layers{
		conv.New(
			conv.FilterSize(3),
			conv.FiltersCount(16),
			conv.Padding(1),
		),
		activation.NewReLu(),
		maxpooling.New(
			maxpooling.FilterSize(2),
			maxpooling.Stride(2),
		),

		conv.New(
			conv.FilterSize(3),
			conv.FiltersCount(16),
			conv.Padding(1),
		),
		activation.NewReLu(),
		maxpooling.New(
			maxpooling.FilterSize(2),
			maxpooling.Stride(2),
		),
		fc.New(fc.OutputSizes(100, 1, 1)),
		softmax.New(),
	})
}
