package layer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/initializer"
	"github.com/atkhx/nnet/num"
)

func NewMSAMultiHead(
	featuresCount int,
	headSize int,
	headsCount int,
	dropoutProb float32,
	initWeights initializer.Initializer,
) *MSAMultiHead {
	heads := make([]nnet.Layer, headsCount)
	for i := 0; i < headsCount; i++ {
		heads[i] = NewMSAHead(featuresCount, headSize, dropoutProb, initWeights)
	}
	return &MSAMultiHead{Heads: heads}
}

type MSAMultiHead struct {
	Heads Layers
}

func (l *MSAMultiHead) Compile(device nnet.Device, inputs *num.Data) *num.Data {
	if len(l.Heads) == 1 {
		return l.Heads[0].Compile(device, inputs)
	}

	var headObjects []*num.Data
	for _, head := range l.Heads {
		headObjects = append(headObjects, head.Compile(device, inputs))
	}
	return device.ConcatByRows(headObjects...)
}

func (l *MSAMultiHead) ForUpdate() []*num.Data {
	return l.Heads.ForUpdate()
}
