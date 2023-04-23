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

	Weights num.Float64s // (storable)
}

func (l *Embed) Compile(bSize int, inputs *num.Data) *num.Data {
	inputsLen := len(inputs.GetData())

	l.iSize = inputsLen / bSize
	l.bSize = bSize

	embedSize := l.featuresCount * l.alphabetSize
	outputSize := l.featuresCount * l.bSize * l.iSize

	{ // code embedding table initialization
		l.Weights = num.NewFloat64sRandNorm(embedSize)
		l.embedObj = num.Wrap(l.Weights, num.NewFloat64s(embedSize))
	}

	l.inputsObj = inputs
	l.outputObj = num.New(outputSize)

	return l.outputObj
}

func (l *Embed) Forward() {
	l.embedObj.GetEmbeddedTo(l.outputObj, l.featuresCount, l.inputsObj.GetData().ToInt())
}

func (l *Embed) ForUpdate() num.Nodes {
	return num.Nodes{l.embedObj}
}
