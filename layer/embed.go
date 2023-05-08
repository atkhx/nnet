package layer

import (
	"github.com/atkhx/nnet/num"
)

func NewEmbed(
	featuresCount int,
	alphabetSize int,
) *Embed {
	return &Embed{
		Embedded: num.NewRandNorm(num.NewDims(
			featuresCount,
			alphabetSize,
		)),
	}
}

type Embed struct {
	Embedded  *num.Data
	outputObj *num.Data
}

func (l *Embed) Compile(inputs *num.Data) *num.Data {
	l.outputObj = l.Embedded.GetEmbedded(inputs)
	return l.outputObj
}

func (l *Embed) Forward() {
	l.outputObj.Forward()
}

func (l *Embed) ForUpdate() num.Nodes {
	return num.Nodes{l.Embedded}
}
