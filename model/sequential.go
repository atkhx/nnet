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

func NewSequential(inDims num.Dims, layers []layer.Layer) *Sequential {
	return &Sequential{
		inDims: inDims,
		Layers: layers,
	}
}

type Sequential struct {
	inDims num.Dims
	inputs *num.Data
	output *num.Data
	Layers layer.Layers
	update num.Nodes
}

func (s *Sequential) Compile() *num.Data {
	s.inputs = num.New(s.inDims)
	s.output = s.Layers.Compile(s.inputs)

	for _, node := range s.Layers.ForUpdate() {
		s.update = append(s.update, node)
	}

	return s.output
}

func (s *Sequential) Forward(inputs num.Float64s) *num.Data {
	copy(s.inputs.Data, inputs)
	s.Layers.Forward()
	return s.output
}

func (s *Sequential) Backward() {
	s.Layers.Backward()
}

func (s *Sequential) GetTrainableParamsCount() int {
	var result int
	for _, node := range s.update {
		result += len(node.Data)
	}
	return result
}

func (s *Sequential) Update(learningRate float64) {
	for _, node := range s.update {
		for j, g := range node.Grad {
			node.Data[j] -= g * learningRate
		}
	}
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
