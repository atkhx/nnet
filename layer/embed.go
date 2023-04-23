package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewEmbed(
	featuresCount int,
	alphabetSize int,
) *Embed {
	return &Embed{
		alphabetSize:  alphabetSize,
		featuresCount: featuresCount,
	}
}

type Embed struct {
	featuresCount int
	alphabetSize  int

	iSize int
	bSize int

	embedObj  *num.Data
	inputsObj *num.Data
	outputObj *num.Data

	// internal buffers
	Weights num.Float64s // (storable)
}

func (l *Embed) Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	l.iSize = len(inputs) / bSize
	l.bSize = bSize

	outputSize := l.featuresCount * l.bSize * l.iSize

	{ // code embedding table initialization
		codeEmbeddingSize := l.featuresCount * l.alphabetSize

		l.Weights = num.NewFloat64sRandNorm(codeEmbeddingSize)
		l.embedObj = num.Wrap(l.Weights, num.NewFloat64s(codeEmbeddingSize))
	}

	l.inputsObj = num.Wrap(inputs, iGrads)

	// candidate to clever output object
	output := num.NewFloat64s(outputSize)
	oGrads := num.NewFloat64s(outputSize)
	l.outputObj = num.Wrap(output, oGrads)

	return output, oGrads
}

func (l *Embed) Forward() {
	l.embedObj.GetEmbeddedTo(l.outputObj, l.featuresCount, l.inputsObj.GetData().ToInt())
}

func (l *Embed) Backward() {
	l.outputObj.CalcGrad()
}

func (l *Embed) ResetGrads() {
	l.outputObj.ResetGrad()
}

func (l *Embed) ForUpdate() num.Nodes {
	return num.Nodes{l.embedObj}
}
