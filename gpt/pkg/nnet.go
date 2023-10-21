package pkg

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/model"
	"github.com/atkhx/nnet/num"
)

const (
	MiniBatchSize = 10

	ContextLength  = 64
	HeadSize       = 64
	HeadsCount     = 8
	FeaturesCount  = HeadSize * HeadsCount
	HeadLinearSize = 4
	BlocksCount    = 1
	DropoutProb    = 0.3
)

func CreateOptimizer(epochs int, device nnet.Device) func(nodes []*num.Data) func(iteration int) {
	return device.GetOptimizerAdam(epochs, 0.9, 0.98, 0.0003, 0.000000001)
}

func CreateModel(
	alphabetSize int,
	miniBatchSize int,
	device nnet.Device,
	optimizer model.Optimizer,
) *model.Sequential {
	return model.NewTransformer(
		ContextLength,
		FeaturesCount,
		HeadsCount,
		HeadSize,
		HeadLinearSize,
		BlocksCount,

		alphabetSize,
		miniBatchSize,
		DropoutProb,
		device,
		optimizer,
	)
}
