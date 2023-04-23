package layer

import "github.com/atkhx/nnet/num"

func NewBias() *Bias {
	return &Bias{}
}

type Bias struct {
	iSize int
	bSize int

	// clever objects
	weightObj *num.Data
	inputsObj *num.Data
	outputObj *num.Data

	Weights num.Float64s // (storable)
}

func (l *Bias) Compile(bSize int, inputs *num.Data) *num.Data {
	inputsLen := len(inputs.GetData())

	l.iSize = inputsLen / bSize
	l.bSize = bSize

	l.Weights = num.NewFloat64s(l.iSize)
	l.weightObj = num.Wrap(l.Weights, num.NewFloat64s(l.iSize))

	l.inputsObj = inputs
	l.outputObj = num.New(inputsLen)

	return l.outputObj
}

func (l *Bias) Forward() {
	l.inputsObj.AddTo(l.outputObj, l.weightObj)
}

func (l *Bias) ForUpdate() num.Nodes {
	return num.Nodes{l.weightObj}
}
