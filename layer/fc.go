package layer

import (
	"fmt"
	"strings"

	"github.com/atkhx/nnet/num"
)

func NewFC(dims num.Dims, initWeights InitWeights) *FC {
	return &FC{dims: dims, initWeights: initWeights}
}

type FC struct {
	initWeights InitWeights

	dims num.Dims

	WeightObj *num.Data
	outputObj *num.Data
	forUpdate num.Nodes
}

func (l *FC) Compile(inputs *num.Data) *num.Data {
	weightK := l.initWeights.GetNormK(len(inputs.Data))

	l.WeightObj = num.NewRandNormWeighted(l.dims, weightK)
	l.outputObj = inputs.MatrixMultiply(l.WeightObj)
	l.forUpdate = num.Nodes{l.WeightObj}

	fmt.Println(strings.Repeat("-", 40))
	fmt.Println("FC\t", l.WeightObj.Dims, "out", l.outputObj.Dims)

	return l.outputObj
}

func (l *FC) Forward() {
	l.outputObj.Forward()
}

func (l *FC) Backward() {
	l.outputObj.Backward()
}

func (l *FC) ForUpdate() num.Nodes {
	return l.forUpdate
}
