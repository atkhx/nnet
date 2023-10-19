package layer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/initializer"
)

func NewMSAMultiHead[data any](
	featuresCount int,
	headSize int,
	headsCount int,
	dropoutProb float32,
	initWeights initializer.Initializer,
) *MSAMultiHead[data] {
	heads := make([]nnet.Layer[data], headsCount)
	for i := 0; i < headsCount; i++ {
		heads[i] = NewMSAHead[data](featuresCount, headSize, dropoutProb, initWeights)
	}
	return &MSAMultiHead[data]{Heads: heads}
}

type MSAMultiHead[data any] struct {
	Heads Layers[data]
}

func (l *MSAMultiHead[data]) Compile(device nnet.Device[data], inputs data) data {
	if len(l.Heads) == 1 {
		return l.Heads[0].Compile(device, inputs)
	}

	var headObjects []data
	for _, head := range l.Heads {
		headObjects = append(headObjects, head.Compile(device, inputs))
	}
	return device.ConcatByRows(headObjects...)
}

func (l *MSAMultiHead[data]) ForUpdate() []data {
	return l.Heads.ForUpdate()
}
