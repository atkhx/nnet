package dataset

import "github.com/atkhx/nnet/num"

type Dataset interface {
	GetSamplesCount() int

	GetLabels() []string
	GetTargets() []*num.Data

	GetLabel(index int) (string, error)
	GetTarget(index int) (*num.Data, error)

	ReadSample(index int) (input, target *num.Data, err error)
	ReadRandomSampleBatch(batchSize int) (input, target *num.Data, err error)
}
