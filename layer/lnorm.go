package layer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/num"
)

func NewLNorm[data any]() *LNorm[data] {
	return &LNorm[data]{}
}

type LNorm[data any] struct {
	Gamma data
	Beta  data

	forUpdate []data
}

func (l *LNorm[data]) Compile(device nnet.Device[data], inputs data) data {
	rowWidth := device.GetDataDims(inputs).W

	l.Beta = device.NewData(num.NewDims(rowWidth))
	l.Gamma = device.NewData(num.NewDims(rowWidth))

	device.FillDataWithOnes(l.Gamma)

	l.forUpdate = []data{l.Gamma, l.Beta}

	return device.LNorm(inputs, l.Gamma, l.Beta)
}

func (l *LNorm[data]) ForUpdate() []data {
	return l.forUpdate
}
