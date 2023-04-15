package block

import "github.com/atkhx/nnet/model"

func NewSequentialBlock(layers []model.Layer) *Sequential {
	return &Sequential{layers: layers}
}

type Sequential struct {
	inputs []float64
	iGrads []float64

	output []float64
	oGrads []float64

	layers []model.Layer
}

func (s *Sequential) Compile(inputs, iGrads []float64) ([]float64, []float64) {
	s.inputs = inputs
	s.iGrads = iGrads

	for _, layer := range s.layers {
		inputs, iGrads = layer.Compile(inputs, iGrads)
	}

	s.output = inputs
	s.oGrads = iGrads

	return s.output, s.oGrads
}

func (s *Sequential) Forward() {
	for _, layer := range s.layers {
		layer.Forward()
	}
}

func (s *Sequential) Backward() {
	for i := len(s.layers); i > 0; i-- {
		s.layers[i-1].Backward()
	}
}

func (s *Sequential) ResetGrads() {
	for _, layer := range s.layers {
		if l, ok := layer.(model.WithGrads); ok {
			l.ResetGrads()
		}
	}
}

func (s *Sequential) ForUpdate() [][2][]float64 {
	result := make([][2][]float64, 0, len(s.layers))

	for _, layer := range s.layers {
		if l, ok := layer.(model.Updatable); ok {
			result = append(result, l.ForUpdate()...)
		}
	}
	return result
}
