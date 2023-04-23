package block

import (
	"github.com/atkhx/nnet/layer"
	"github.com/atkhx/nnet/num"
)

func NewSequentialBlock(layers layer.Layers) *Sequential {
	return &Sequential{Layers: layers}
}

type Sequential struct {
	inputs *num.Data
	output *num.Data

	Layers layer.Layers
}

func (s *Sequential) Compile(bSize int, inputs *num.Data) *num.Data {
	s.inputs = inputs
	s.output = s.Layers.Compile(bSize, inputs)

	return s.output
}

func (s *Sequential) Forward() {
	s.Layers.Forward()
}

func (s *Sequential) ForUpdate() num.Nodes {
	return s.Layers.ForUpdate()
}
