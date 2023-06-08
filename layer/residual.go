package layer

import "github.com/atkhx/nnet/num"

func NewResidual(layers Layers) *Residual {
	return &Residual{Layers: layers}
}

type Residual struct {
	Layers    Layers
	inputsObj *num.Data
	outputPre *num.Data
	outputObj *num.Data
}

func (l *Residual) Compile(inputs *num.Data) *num.Data {
	l.inputsObj = inputs
	l.outputPre = l.Layers.Compile(inputs)
	l.outputObj = l.outputPre.Add(inputs)

	return l.outputObj
}

func (l *Residual) ForUpdate() num.Nodes {
	return l.Layers.ForUpdate()
}

func (l *Residual) GetInputs() *num.Data {
	return l.inputsObj
}

func (l *Residual) GetOutput() *num.Data {
	return l.outputObj
}
