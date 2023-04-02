package pkg

import (
	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/dataset/mnist"
	"github.com/atkhx/nnet/layer/activation"
	"github.com/atkhx/nnet/layer/conv"
	"github.com/atkhx/nnet/layer/fc"
	"github.com/atkhx/nnet/layer/maxpooling"
	"github.com/atkhx/nnet/layer/reshape"
	"github.com/atkhx/nnet/net"
)

func CreateConvNet(batchSize int) *net.FeedForward {
	return net.New(
		net.Layers{
			conv.New(
				conv.WithInputSize(
					mnist.ImageWidth,
					mnist.ImageHeight,
					mnist.ImageDepth,
				),
				conv.WithFilterSize(5),
				conv.WithFiltersCount(16),
				conv.WithPadding(2),
				conv.WithBatchSize(batchSize),
				conv.WithGain(data.ReLuGain),
			),

			activation.NewReLu(),

			maxpooling.New(
				maxpooling.WithInputSize(
					28,
					28,
					16,
				),
				maxpooling.FilterSize(2),
				maxpooling.Stride(2),
			),

			conv.New(
				conv.WithInputSize(14, 14, 16),
				conv.WithFilterSize(3),
				conv.WithFiltersCount(10),
				conv.WithPadding(1),
				conv.WithBatchSize(batchSize),
				conv.WithGain(data.ReLuGain),
			),
			activation.NewReLu(),

			reshape.New(func(iw, ih, id int) (int, int, int) {
				return iw * ih, id, 1
			}),

			fc.New(
				fc.WithInputSize(14*14*10),
				fc.WithLayerSize(10),
				fc.WithBiases(true),
				fc.WithBatchSize(batchSize),
				fc.WithGain(data.LinearGain),
			),
		},
	)
}
