package pkg

import (
	"context"

	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/model"
	"github.com/atkhx/nnet/num"
)

const (
	MiniBatchSize = 55

	ContextLength = 512
	FeaturesCount = 128
	HeadsCount    = 16
	HeadSize      = FeaturesCount / HeadsCount
	KVHeadsCount  = 1
	HiddenDim     = 768 //4 * FeaturesCount
	BlocksCount   = 2

	DropoutProb = 0
	InitWeightK = 0.007

	adamBeta1        = 0.9
	adamBeta2        = 0.98
	adamLearningRate = 0.0003
	adamEPS          = 0.000000001
)

func CreateOptimizer(epochs int, device nnet.Device) func(nodes []*num.Data) func(ctx context.Context, iteration int) {
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
		KVHeadsCount,
		HeadSize,
		HiddenDim,
		BlocksCount,

		alphabetSize,
		miniBatchSize,
		DropoutProb,
		InitWeightK,

		device,
		optimizer,
	)
}

func CreateModelForTest(
	alphabetSize int,
	miniBatchSize int,
	device nnet.Device,
) *model.Sequential {
	return model.NewLLaMaExperiment(
		//return model.NewTransformer(
		ContextLength,
		FeaturesCount,
		HeadsCount,
		KVHeadsCount,
		HeadSize,
		HiddenDim,
		BlocksCount,

		alphabetSize,
		miniBatchSize,
		0,
		InitWeightK,

		device,
		nil,
	)
}
