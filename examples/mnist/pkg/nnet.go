package pkg

import (
	"github.com/atkhx/nnet/dataset/mnist"
	"github.com/atkhx/nnet/layer/activation"
	"github.com/atkhx/nnet/layer/conv"
	"github.com/atkhx/nnet/layer/fc"
	"github.com/atkhx/nnet/layer/flat"
	"github.com/atkhx/nnet/layer/softmax"
	"github.com/atkhx/nnet/net"
)

func CreateConvNet() *net.FeedForward {
	return CreateConvNetOneLayer()
	return CreateConvNetTwoLayer()
}

func CreateConvNetTwoLayer() *net.FeedForward {
	return net.New(
		net.Layers{
			conv.New(
				conv.WithInputSize(
					mnist.ImageWidth,
					mnist.ImageHeight,
					mnist.ImageDepth,
				),
				conv.WithFilterSize(3),
				conv.WithFiltersCount(16),
				conv.WithPadding(0),
				// out: 26 x 26 x 16
			),
			activation.NewReLu(),

			conv.New(
				conv.WithInputSize(26, 26, 16),
				conv.WithFilterSize(3),
				conv.WithFiltersCount(3),
				// out: 24 x 24 x 3
			),
			activation.NewReLu(),

			flat.New(), // todo respect the batches!

			fc.New(
				fc.WithInputSize(24*24*3),
				fc.WithLayerSize(10),
				fc.WithBiases(true),
			),
			softmax.New(),
			// panic: invalid targets dimensions: expected [11 24], actual [10 1]
		},
	)
}

func CreateConvNetOneLayer() *net.FeedForward {
	return net.New(
		net.Layers{
			conv.New(
				conv.WithInputSize(
					mnist.ImageWidth,
					mnist.ImageHeight,
					mnist.ImageDepth,
				),
				conv.WithFilterSize(3),
				conv.WithFiltersCount(16),
				conv.WithPadding(0),
				// out: 26 x 26 x 16
			),
			activation.NewReLu(),

			flat.New(), // todo respect the batches!

			fc.New(
				fc.WithInputSize(26*26*16),
				fc.WithLayerSize(10),
				fc.WithBiases(true),
			),
			softmax.New(),
		},
	)
}

func CreateNetFC() *net.FeedForward {
	return net.New(
		net.Layers{
			flat.New(),
			fc.New(
				fc.WithInputSize(28*28*1),
				fc.WithLayerSize(20),
				fc.WithBiases(true),
			),
			//activation.NewReLu(),
			activation.NewTanh(),
			fc.New(
				fc.WithInputSize(20),
				fc.WithLayerSize(10),
				fc.WithBiases(true),
			),
			//activation.NewReLu(),
			//activation.NewTanh(),

			flat.New(),

			softmax.New(),
		},
	)
}
