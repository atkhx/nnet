package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewLNorm() *LNorm {
	return &LNorm{}
}

type LNorm struct {
	Gamma     *num.Data
	Beta      *num.Data
	inputsObj *num.Data
	outputObj *num.Data
	update    num.Nodes
}

func (l *LNorm) Compile(inputs *num.Data) *num.Data {
	l.Gamma = num.New(num.NewDims(inputs.Dims.W))
	l.Gamma.Data.Fill(1)

	l.Beta = num.New(num.NewDims(inputs.Dims.W))
	l.inputsObj = inputs
	l.outputObj = inputs.LNorm(l.Gamma, l.Beta)

	l.update = num.Nodes{l.Gamma, l.Beta}
	return l.outputObj
}

func (l *LNorm) Forward() {
	l.outputObj.Forward()
}

func (l *LNorm) Backward() {
	l.outputObj.Backward()
}

func (l *LNorm) ForUpdate() num.Nodes {
	return l.update
}

func (l *LNorm) GetInputs() *num.Data {
	return l.inputsObj
}

func (l *LNorm) GetOutput() *num.Data {
	return l.outputObj
}
