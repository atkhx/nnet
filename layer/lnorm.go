package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewLNorm() *LNorm {
	return &LNorm{}
}

type LNorm struct {
	Gamma *num.Data
	Beta  *num.Data

	inputsObj *num.Data
	outputObj *num.Data
	forUpdate num.Nodes
}

func (l *LNorm) Compile(inputs *num.Data) *num.Data {
	l.Gamma = num.New(num.NewDims(inputs.Dims.W))
	l.Gamma.Data.Ones()

	l.Beta = num.New(num.NewDims(inputs.Dims.W))
	l.inputsObj = inputs
	l.outputObj = inputs.LNorm(l.Gamma, l.Beta)
	l.forUpdate = num.Nodes{l.Gamma, l.Beta}

	return l.outputObj
}

func (l *LNorm) ForUpdate() num.Nodes {
	return l.forUpdate
}

func (l *LNorm) GetInputs() *num.Data {
	return l.inputsObj
}

func (l *LNorm) GetOutput() *num.Data {
	return l.outputObj
}
