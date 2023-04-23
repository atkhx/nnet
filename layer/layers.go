package layer

import "github.com/atkhx/nnet/num"

type Layers []Layer

func (s Layers) Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	for _, layer := range s {
		inputs, iGrads = layer.Compile(bSize, inputs, iGrads)
	}

	return inputs, iGrads
}

func (s Layers) Forward() {
	for _, layer := range s {
		layer.Forward()
	}
}

func (s Layers) Backward() {
	for i := len(s); i > 0; i-- {
		s[i-1].Backward()
	}
}

func (s Layers) ResetGrads() {
	for _, layer := range s {
		if l, ok := layer.(WithGrads); ok {
			l.ResetGrads()
		}
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
