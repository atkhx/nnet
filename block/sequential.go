package block

import (
	"github.com/atkhx/nnet/layer"
)

func NewSequentialBlock(layers layer.Layers) *Sequential {
	return &Sequential{layers: layers}
}

type Sequential struct {
	inputs []float64
	iGrads []float64

	output []float64
	oGrads []float64

	layers layer.Layers
}

func (s *Sequential) Compile(inputs, iGrads []float64) ([]float64, []float64) {
	s.inputs = inputs
	s.iGrads = iGrads

	s.output, s.oGrads = s.layers.Compile(inputs, iGrads)

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

func (s *Sequential) ForUpdate() [][2][]float64 {
	return s.layers.ForUpdate()
}
