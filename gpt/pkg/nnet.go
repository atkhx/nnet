package pkg

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/model"
	"github.com/atkhx/nnet/num"
)

const (
	MiniBatchSize = 10

	ContextLength  = 512
	HeadSize       = 64
	HeadsCount     = 16
	FeaturesCount  = HeadSize * HeadsCount
	HeadLinearSize = 4
	BlocksCount    = 1
	DropoutProb    = 0.1
	InitWeightK    = 0.007

	adamBeta1        = 0.9
	adamBeta2        = 0.8
	adamLearningRate = 0.0003
	adamEPS          = 0.000000001
)

func CreateOptimizer(epochs int, device nnet.Device) func(nodes []*num.Data) func(iteration int) {
	return device.GetOptimizerAdam(epochs, adamBeta1, adamBeta2, adamLearningRate, adamEPS)
}

func CreateModel(
	alphabetSize int,
	miniBatchSize int,
	device nnet.Device,
	optimizer model.Optimizer,
) *model.Sequential {
	return model.NewLLaMaExperiment(
		//return model.NewTransformer(
		ContextLength,
		FeaturesCount,
		HeadsCount,
		HeadSize,
		HeadLinearSize,
		BlocksCount,

		alphabetSize,
		miniBatchSize,
		DropoutProb,
		InitWeightK,

		device,
		optimizer,
	)
}
