package dataset

import "github.com/atkhx/nnet/num"

type Sample struct {
	Input, Target *num.Data
}

type Dataset interface {
	GetSamplesCount() int
	ReadSample(index int) (sample Sample, err error)
	ReadRandomSampleBatch(batchSize int) (sample Sample, err error)
}

type ClassifierDataset interface {
	Dataset

	GetClasses() []string
}
