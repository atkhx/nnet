package layer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/num"
)

func NewLinear[data any](featuresCount int, initWeights initializer.Initializer) *Linear[data] {
	return &Linear[data]{featuresCount: featuresCount, initWeights: initWeights}
}

type Linear[data any] struct {
	initWeights   initializer.Initializer
	featuresCount int

	WeightObj data
	BiasesObj data

	forUpdate []data
}

func (l *Linear[data]) Compile(device nnet.Device[data], inputs data) data {
	weightK := l.initWeights.GetNormK(device.GetDataLength(inputs))
	inputWidth := device.GetDataDims(inputs).W

	l.WeightObj = device.NewDataRandNormWeighted(num.NewDims(l.featuresCount, inputWidth), weightK)
	l.BiasesObj = device.NewData(num.NewDims(l.featuresCount))
	l.forUpdate = []data{l.WeightObj, l.BiasesObj}

	return device.Add(device.MatrixMultiply(inputs, l.WeightObj), l.BiasesObj)
}

func (l *Linear[data]) ForUpdate() []data {
	return l.forUpdate
}
