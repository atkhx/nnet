package pkg

import (
	"math"

	"github.com/atkhx/metal/nn/model"
	"github.com/atkhx/metal/nn/proc"
)

const (
	MiniBatchSize      = 10
	TrainingIterations = 5000

	ContextLength = 1024
	FeaturesCount = 1024
	HeadsCount    = 16
	HeadSize      = FeaturesCount / HeadsCount
	HiddenDim     = 2 * FeaturesCount
	BlocksCount   = 2

	DropoutProb = 0

	adamBeta1        = 0.9
	adamBeta2        = 0.98
	adamLearningRate = 0.0003
	adamEPS          = 0.000000001
)

var InitWeightK = float32(1. / math.Sqrt(float64(FeaturesCount/2)))

func CreateOptimizer(epochs int, device *proc.Device) proc.Optimizer {
	return device.GetOptimizerAdam(epochs, adamBeta1, adamBeta2, adamLearningRate, adamEPS)
}

func CreateTrainingModel(
	alphabetSize int,
	miniBatchSize int,
	device *proc.Device,
	optimizer proc.Optimizer,
) *model.Model {
	return NewLLaMaExperiment(
		ContextLength,
		FeaturesCount,
		HeadsCount,
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

func CreateInferenceModel(
	alphabetSize int,
	miniBatchSize int,
	device *proc.Device,
) *model.Model {
	return NewLLaMaExperiment(
		ContextLength,
		FeaturesCount,
		HeadsCount,
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
