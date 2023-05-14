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
	outputObj *num.Data
}

func (l *LNorm) Compile(inputs *num.Data) *num.Data {
	l.Gamma = num.New(num.NewDims(1, inputs.Dims.H))
	l.Gamma.Data.Fill(1)

	l.Beta = num.New(num.NewDims(1, inputs.Dims.H))
	l.outputObj = inputs.LNorm(l.Gamma, l.Beta)

	return l.outputObj
}

func (l *LNorm) Forward() {
	l.outputObj.Forward()
}

func (l *LNorm) Backward() {
	l.outputObj.Backward()
}

func (l *LNorm) ForUpdate() num.Nodes {
	return num.Nodes{l.Gamma, l.Beta}
}
