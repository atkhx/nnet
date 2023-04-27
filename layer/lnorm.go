package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewLNorm() *LNorm {
	return &LNorm{}
}

type LNorm struct {
	iSize int
	bSize int

	// clever objects
	gammaObj *num.Data
	betaObj  *num.Data

	inputsObj *num.Data
	outputObj *num.Data

	Gamma num.Float64s // (storable)
	Beta  num.Float64s // (storable)

	meanBuffer         *num.Data
	varianceBuffer     *num.Data
	sqrtVarianceBuffer *num.Data
	xsubMeanBuffer     *num.Data
	xhatBuffer         *num.Data
	xhatMulBuffer      *num.Data
}

func (l *LNorm) Compile(bSize int, inputs *num.Data) *num.Data {
	inputsLen := len(inputs.GetData())

	l.iSize = inputsLen / bSize
	l.bSize = bSize

	l.Gamma = num.NewFloat64s(l.bSize)
	l.Gamma.Fill(1.0)

	l.gammaObj = num.Wrap(l.Gamma, num.NewFloat64s(l.bSize))

	l.Beta = num.NewFloat64s(l.bSize)
	l.betaObj = num.Wrap(l.Beta, num.NewFloat64s(l.bSize))

	l.inputsObj = inputs
	l.outputObj = num.New(inputsLen)

	l.meanBuffer = num.New(bSize)
	l.varianceBuffer = num.New(bSize)
	l.sqrtVarianceBuffer = num.New(bSize)
	l.xsubMeanBuffer = num.New(inputsLen)
	l.xhatBuffer = num.New(inputsLen)
	l.xhatMulBuffer = num.New(inputsLen)

	return l.outputObj
}

func (l *LNorm) Forward() {
	l.inputsObj.MeanTo(l.meanBuffer, l.bSize)
	l.inputsObj.VarianceTo(l.varianceBuffer, l.bSize, l.meanBuffer.Data)
	l.varianceBuffer.SqrtTo(l.sqrtVarianceBuffer)

	l.inputsObj.SubColVectorTo(l.xsubMeanBuffer, l.bSize, l.meanBuffer)
	l.xsubMeanBuffer.DivColVectorTo(l.xhatBuffer, l.bSize, l.sqrtVarianceBuffer)
	l.xhatBuffer.MulColVectorTo(l.xhatMulBuffer, l.bSize, l.gammaObj)
	l.xhatMulBuffer.AddColVectorTo(l.outputObj, l.bSize, l.betaObj)
}

func (l *LNorm) ForUpdate() num.Nodes {
	return num.Nodes{l.gammaObj, l.betaObj}
}
