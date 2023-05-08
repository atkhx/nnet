package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewEmbedPos(
	featuresCount int,
	alphabetSize int,
	contextSize int,
) *EmbedPos {
	return &EmbedPos{
		Embedded: num.NewRandNorm(num.NewDims(
			featuresCount,
			alphabetSize,
		)),
		Positions: num.NewRandNorm(num.NewDims(
			featuresCount,
			contextSize,
		)),
	}
}

type EmbedPos struct {
	Embedded  *num.Data
	Positions *num.Data

	tFeatures *num.Data
	pFeatures *num.Data
	outputObj *num.Data
}

func (l *EmbedPos) Compile(inputs *num.Data) *num.Data {
	l.tFeatures = l.Embedded.GetEmbedded(inputs)
	l.pFeatures = l.Positions.GetEmbeddedPos(inputs)

	l.outputObj = l.tFeatures.Add(l.pFeatures)
	return l.outputObj
}

func (l *EmbedPos) Forward() {
	l.tFeatures.Forward()
	l.pFeatures.Forward()

	l.outputObj.Forward()
}

func (l *EmbedPos) ForUpdate() num.Nodes {
	return num.Nodes{l.Embedded, l.Positions}
}
