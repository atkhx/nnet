package layer

import "github.com/atkhx/nnet/num"

func NewResidual(layers Layers) *Residual {
	return &Residual{Layers: layers}
}

type Residual struct {
	Layers    Layers
	outputPre *num.Data
	outputObj *num.Data
}

func (l *Residual) Compile(inputs *num.Data) *num.Data {
	l.outputPre = l.Layers.Compile(inputs)
	l.outputObj = l.outputPre.Add(inputs)

	return l.outputObj
}

func (l *Residual) Forward() {
	l.Layers.Forward()
	l.outputObj.Forward()
}

func (l *Residual) Backward() {
	l.outputObj.Backward()
	l.Layers.Backward()
}

func (l *Residual) ForUpdate() num.Nodes {
	return l.Layers.ForUpdate()
}
