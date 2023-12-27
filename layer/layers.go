package layer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/num"
)

type Layers []nnet.Layer

func (s Layers) Compile(device nnet.Device, inputs *num.Data) *num.Data {
	for _, layer := range s {
		inputs = layer.Compile(device, inputs)
	}
	return inputs
}

func (s Layers) ForUpdate() []*num.Data {
	result := make([]*num.Data, 0, len(s))
	for _, layer := range s {
		if l, ok := layer.(nnet.LayerUpdatable); ok {
			result = append(result, l.ForUpdate()...)
		}
	}
	return result
}

func (s Layers) LoadFromProvider() {
	for _, ll := range s {
		if l, ok := ll.(nnet.LayerWithWeightsProvider); ok {
			l.LoadFromProvider()
		}
	}
}
