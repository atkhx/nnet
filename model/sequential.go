package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"

	"github.com/atkhx/nnet/layer"
	"github.com/atkhx/nnet/num"
)

func NewSequential(iSize, bSize int, layers []layer.Layer) *Sequential {
	return &Sequential{
		iSize:  iSize,
		bSize:  bSize,
		Layers: layers,
	}
}

type Sequential struct {
	iSize int
	bSize int

	inputs num.Float64s
	iGrads num.Float64s

	output num.Float64s
	oGrads num.Float64s

	Layers layer.Layers
}

func (s *Sequential) Compile() {
	s.inputs = make(num.Float64s, s.iSize*s.bSize)
	s.iGrads = make(num.Float64s, s.iSize*s.bSize)

	s.output, s.oGrads = s.Layers.Compile(s.bSize, s.inputs, s.iGrads)
}

func (s *Sequential) NewOutput() num.Float64s {
	res := s.output.Copy()
	res.Fill(0)
	return res
}

func (s *Sequential) Forward(inputs, output num.Float64s) {
	copy(s.inputs, inputs)
	s.Layers.Forward()
	copy(output, s.output)
}

func (s *Sequential) Backward(target num.Float64s) {
	chunkSize := len(s.output) / s.bSize

	softmax := s.output.Copy()
	for i := 0; i < len(s.output); i += chunkSize {
		softmax[i : i+chunkSize].Softmax()
	}

	k := 1.0 / float64(s.bSize)
	for i, t := range target {
		s.oGrads[i] = k * (softmax[i] - t)
	}
	s.Layers.Backward()
}

func (s *Sequential) Update(learningRate float64) {
	for _, pair := range s.Layers.ForUpdate() {
		for j := range pair[1] {
			pair[0][j] -= pair[1][j] * learningRate
		}
	}

	s.Layers.ResetGrads()
}

func (s *Sequential) LoadFromFile(filename string) error {
	config, err := os.ReadFile(filename)
	if err != nil && errors.Is(err, os.ErrNotExist) {
		log.Println("trained config not found (skip)")
		return nil
	}

	if err != nil {
		return fmt.Errorf("read file failed: %w", err)
	}

	if err = json.Unmarshal(config, s); err != nil {
		return fmt.Errorf("unmarshal config failed: %w", err)
	}
	return nil
}

func (s *Sequential) SaveToFile(filename string) error {
	nnBytes, err := json.Marshal(s)
	if err != nil {
		return fmt.Errorf("marshal model config failed: %w", err)
	}

	if err := os.WriteFile(filename, nnBytes, os.ModePerm); err != nil {
		return fmt.Errorf("write model config failed: %w", err)
	}
	return nil
}
