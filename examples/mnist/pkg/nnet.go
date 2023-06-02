package pkg

import (
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/layer"
	"github.com/atkhx/nnet/model"
	"github.com/atkhx/nnet/num"
	"github.com/atkhx/nnet/optimizer"
)

func CreateConvNet(batchSize int) *model.Sequential {
	return model.NewSequential(
		num.NewDims(28*28, 1, batchSize),
		layer.Layers{
			layer.NewLNorm(),
			layer.NewConv(3, 10, 1, 1, 28, 28, initializer.KaimingNormalReLU), // todo -> out: 26x26, 1, 1
			layer.NewReLu(),
			// out: [ 28x28, 10, 1 ]

			layer.NewMaxPooling(28, 28, 2, 0, 2),
			// out: [ 14x14, 10, 1 ]

			layer.NewLNorm(),
			layer.NewConv(3, 10, 1, 1, 14, 14, initializer.KaimingNormalReLU), // todo -> out: 26x26, 1, 1
			layer.NewReLu(),
			// out: [ 14x14, 10, 1 ]

			layer.NewMaxPooling(14, 14, 2, 0, 2),
			// out: [ 7x7, 10, 1 ]

			layer.NewReshape(num.NewDims(7*7*10, 1)),
			layer.NewLinear(10, initializer.KaimingNormalReLU),
		},
		optimizer.Adadelta(optimizer.Ro, optimizer.Eps),
	)
}
