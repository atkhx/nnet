package layer

import "github.com/atkhx/nnet/num"

type Layers []Layer

func (s Layers) Compile(inputs *num.Data) *num.Data {
	for _, layer := range s {
		inputs = layer.Compile(inputs)
	}

	return inputs
}

func (s Layers) Forward() {
	for _, layer := range s {
		layer.Forward()
	}
}

func (s Layers) ForUpdate() num.Nodes {
	result := make(num.Nodes, 0, len(s))
	for _, layer := range s {
		if l, ok := layer.(Updatable); ok {
			result = append(result, l.ForUpdate()...)
		}
	}
	return result
}
