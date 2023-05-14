package layer

import (
	"fmt"
	"math"
	"strings"

	"github.com/atkhx/nnet/num"
)

func NewFC(dims num.Dims, gain float64, label string) *FC {
	return &FC{dims: dims, gain: gain, label: label}
}

type FC struct {
	dims  num.Dims
	gain  float64
	label string

	WeightObj *num.Data
	outputObj *num.Data
	forUpdate num.Nodes
}

func (l *FC) Compile(inputs *num.Data) *num.Data {
	weightK := 1.0

	if l.gain > 0 {
		fanIn := len(inputs.Data)
		weightK = l.gain / math.Pow(float64(fanIn), 0.5)
	}

	l.WeightObj = num.NewRandNormWeighted(l.dims, weightK)
	l.outputObj = inputs.MatrixMultiply(l.WeightObj)
	l.outputObj.SetLabel(l.label)
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
