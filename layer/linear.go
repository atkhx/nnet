package layer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/num"
)

func NewLinear(featuresCount int, initWeights initializer.Initializer) *Linear {
	return &Linear{featuresCount: featuresCount, initWeights: initWeights}
}

type Linear struct {
	initWeights   initializer.Initializer
	featuresCount int

	WeightObj *num.Data
	BiasesObj *num.Data

	forUpdate []*num.Data
}

func (l *Linear) Compile(device nnet.Device, inputs *num.Data) *num.Data {
	weightK := l.initWeights.GetNormK(device.GetDataLength(inputs))
	inputWidth := device.GetDataDims(inputs).W

	l.WeightObj = device.NewDataRandNormWeighted(num.NewDims(l.featuresCount, inputWidth), weightK)
	l.BiasesObj = device.NewData(num.NewDims(l.featuresCount))
	l.forUpdate = []*num.Data{l.WeightObj, l.BiasesObj}

	return device.Add(device.MatrixMultiply3D(inputs, l.WeightObj, 1), l.BiasesObj)
}

func (l *Linear) ForUpdate() []*num.Data {
	return l.forUpdate
}
