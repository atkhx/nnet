package model

import (
	"github.com/atkhx/nnet/layer"
)

func NewSequential(iSize, bSize int, layers []layer.Layer) *Sequential {
	return &Sequential{
		iSize:  iSize,
		bSize:  bSize,
		layers: layers,
	}
}

type Sequential struct {
	iSize int
	bSize int

	inputs []float64
	iGrads []float64

	output []float64
	oGrads []float64

	layers layer.Layers
}

func (s *Sequential) Compile() {
	s.inputs = make([]float64, s.iSize*s.bSize)
	s.iGrads = make([]float64, s.iSize*s.bSize)

	s.output, s.oGrads = s.layers.Compile(s.bSize, s.inputs, s.iGrads)
}

func (s *Sequential) Forward(inputs, output []float64) {
	copy(s.inputs, inputs)
	s.layers.Forward()
	copy(output, s.output)
}

func (s *Sequential) Backward(target []float64) {
	k := 1.0 / float64(s.bSize)
	for i, t := range target {
		s.oGrads[i] = k * (s.output[i] - t)
	}

	//fmt.Println("s.oGrads", s.oGrads)
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
