package layer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/num"
)

func NewLinearWithWeights(
	weightObj *num.Data,
) *LinearWithWeights {
	return &LinearWithWeights{
		weightObj: weightObj,
	}
}

type LinearWithWeights struct {
	weightObj *num.Data
}

func (l *LinearWithWeights) Compile(device nnet.Device, inputs *num.Data) *num.Data {
	return device.MatrixMultiply(inputs, l.weightObj, 1)
}
