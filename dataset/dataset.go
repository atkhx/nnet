package dataset

import "github.com/atkhx/nnet/data"

type Dataset interface {
	GetSamplesCount() int

	GetLabels() []string
	GetTargets() []*data.Matrix

	GetLabel(index int) (string, error)
	GetTarget(index int) (*data.Matrix, error)

	ReadSample(index int) (input, target *data.Matrix, err error)
}
