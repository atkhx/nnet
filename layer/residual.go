package layer

import (
	"github.com/atkhx/nnet"
)

func NewResidual[data any](layers Layers[data]) *Residual[data] {
	return &Residual[data]{Layers: layers}
}

type Residual[data any] struct {
	Layers Layers[data]
}

func (l *Residual[data]) Compile(device nnet.Device[data], inputs data) data {
	return device.Add(inputs, l.Layers.Compile(device, inputs))
}

func (l *Residual[data]) ForUpdate() []data {
	return l.Layers.ForUpdate()
}
