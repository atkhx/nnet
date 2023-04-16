package model

import "github.com/atkhx/nnet/layer"

func NewSequential(iSize int, layers []layer.Layer) *Sequential {
	return &Sequential{
		iSize:  iSize,
		layers: layers,
	}
}

type Sequential struct {
	iSize int

	inputs []float64
	iGrads []float64

	output []float64
	oGrads []float64

	layers layer.Layers
}

func (s *Sequential) Compile() {
	s.inputs = make([]float64, s.iSize)
	s.iGrads = make([]float64, s.iSize)

	s.output, s.oGrads = s.layers.Compile(s.inputs, s.iGrads)
}

func (s *Sequential) Forward(inputs, output []float64) {
	copy(s.inputs, inputs)
	s.layers.Forward()
	copy(output, s.output)
}

func (s *Sequential) Backward(target []float64) {
	for i, t := range target {
		s.oGrads[i] = s.output[i] - t
	}

	s.layers.Backward()
}

func (s *Sequential) Update(learningRate float64) {
	for _, pair := range s.layers.ForUpdate() {
		for j := range pair[1] {
			pair[0][j] -= pair[1][j] * learningRate
		}
	}

	s.layers.ResetGrads()
}
