package layer

import (
	"math"

	"github.com/atkhx/nnet/num"
)

func NewFC(dims num.Dims, gain float64) *FC {
	return &FC{dims: dims, gain: gain}
}

type FC struct {
	dims num.Dims
	gain float64

	WeightObj *num.Data
	outputObj *num.Data
}

func (l *FC) Compile(inputs *num.Data) *num.Data {
	weightK := 1.0

	if l.gain > 0 {
		fanIn := len(inputs.Data)
		weightK = l.gain / math.Pow(float64(fanIn), 0.5)
	}

	l.WeightObj = num.NewRandNormWeighted(l.dims, weightK)
	l.outputObj = inputs.MatrixMultiply(l.WeightObj)

	return l.outputObj
}

func (l *FC) Forward() {
	l.outputObj.Forward()
}

func (l *FC) ForUpdate() num.Nodes {
	return num.Nodes{l.WeightObj}
}
