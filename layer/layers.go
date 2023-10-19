package layer

import (
	"github.com/atkhx/nnet"
)

type Layers[data any] []nnet.Layer[data]

func (s Layers[data]) Compile(device nnet.Device[data], inputs data) data {
	for _, layer := range s {
		inputs = layer.Compile(device, inputs)
	}

	return inputs
}

func (s Layers[data]) ForUpdate() []data {
	result := make([]data, 0, len(s))
	for _, layer := range s {
		if l, ok := layer.(nnet.LayerUpdatable[data]); ok {
			result = append(result, l.ForUpdate()...)
		}
	}
	return result
}
