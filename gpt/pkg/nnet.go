package pkg

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/model"
)

const (
	TrainingMiniBatchSize = 10

	ContextLength  = 64
	HeadSize       = 64
	HeadsCount     = 1
	FeaturesCount  = HeadSize * HeadsCount
	HeadLinearSize = 4
	BlocksCount    = 1
	DropoutProb    = 0.3
)

func CreateNN[data any](
	alphabetSize int,
	miniBatchSize int,
	device nnet.Device[data],
	modelOptimizer model.Optimizer[data],
) *model.Sequential[data] {
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
		modelOptimizer,
	)
}
