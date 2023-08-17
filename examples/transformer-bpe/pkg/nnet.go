package pkg

import (
	"github.com/atkhx/nnet/model"
)

const (
	TrainingMiniBatchSize = 10

	ContextLength  = 64
	HeadSize       = 64
	HeadsCount     = 8
	FeaturesCount  = HeadSize * HeadsCount
	HeadLinearSize = 4
	BlocksCount    = 4
	DropoutProb    = 0.1
)

func CreateNN(
	alphabetSize int,
	miniBatchSize int,
	modelOptimizer model.Optimizer,
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
		modelOptimizer,
	)
}
