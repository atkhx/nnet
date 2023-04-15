package block

import "github.com/atkhx/nnet/model"

func NewSequential(inputs, iGrads []float64) *Sequential {
	return &Sequential{
		inputs: inputs,
		iGrads: iGrads,
		output: nil,
		oGrads: nil,
	}
}

type Sequential struct {
	inputs []float64
	iGrads []float64

	output []float64
	oGrads []float64

	layers []model.Layer
}

func (s *Sequential) RegisterLayer(newLayer func(inputs, iGrads []float64) model.Layer) {
	var layer model.Layer
	if len(s.layers) == 0 {
		layer = newLayer(s.inputs, s.iGrads)
	} else {
		layer = newLayer(s.output, s.oGrads)
	}

	s.layers = append(s.layers, layer)
	s.output, s.oGrads = layer.Buffers()
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

func (s *Sequential) ForUpdate() [][2][]float64 {
	var result [][2][]float64
	for _, layer := range s.layers {
		if l, ok := layer.(model.Updatable); ok {
			result = append(result, l.ForUpdate()...)
		}
	}

	return result
}
