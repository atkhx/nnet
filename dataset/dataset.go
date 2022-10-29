package dataset

import "github.com/atkhx/nnet/data"

type Dataset interface {
	GetSamplesCount() int

	GetLabels() []string
	GetTargets() []*data.Data

	GetLabel(index int) (string, error)
	GetTarget(index int) (*data.Data, error)

	ReadSample(index int) (input, target *data.Data, err error)
}
