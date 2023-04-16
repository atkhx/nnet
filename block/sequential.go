package block

import (
	"github.com/atkhx/nnet/layer"
	"github.com/atkhx/nnet/num"
)

func NewSequentialBlock(layers layer.Layers) *Sequential {
	return &Sequential{layers: layers}
}

type Sequential struct {
	inputs num.Float64s
	iGrads num.Float64s

	output num.Float64s
	oGrads num.Float64s

	layers layer.Layers
}

func (s *Sequential) Compile(bSize int, inputs, iGrads num.Float64s) (num.Float64s, num.Float64s) {
	s.inputs = inputs
	s.iGrads = iGrads

	s.output, s.oGrads = s.layers.Compile(bSize, inputs, iGrads)

	return s.output, s.oGrads
}

func (s *Sequential) Forward() {
	s.layers.Forward()
}

func (s *Sequential) Backward() {
	s.layers.Backward()
}

func (s *Sequential) ResetGrads() {
	s.layers.ResetGrads()
}

func (s *Sequential) ForUpdate() [][2]num.Float64s {
	return s.layers.ForUpdate()
}
