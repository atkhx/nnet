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

	inputs *num.Data
	output *num.Data

	Layers layer.Layers
}

func (s *Sequential) Compile() {
	s.inputs = num.New(s.iSize * s.bSize)
	s.output = s.Layers.Compile(s.bSize, s.inputs)
}

func (s *Sequential) NewOutput() num.Float64s {
	res := s.output.GetData().Copy()
	res.Fill(0)
	return res
}

func (s *Sequential) Forward(inputs, output num.Float64s) {
	copy(s.inputs.GetData(), inputs)
	s.Layers.Forward()
	copy(output, s.output.GetData())
}

func (s *Sequential) Backward(oGrads num.Float64s) {
	pair := s.output.ForUpdate()
	copy(pair[1], oGrads)
	s.output.CalcGrad()
}

func (s *Sequential) Update(learningRate float64) {
	for _, node := range s.Layers.ForUpdate() {
		pair := node.ForUpdate()
		for j := range pair[1] {
			pair[0][j] -= pair[1][j] * learningRate
		}
	}

	s.output.ResetGrad()
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
