package model

import (
	"errors"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/layer"
	"github.com/atkhx/nnet/num"

	jsoniter "github.com/json-iterator/go"
)

var json = jsoniter.ConfigCompatibleWithStandardLibrary

type Optimizer[data any] func(nodes []data) func(iteration int)

func NewSequential[data any](
	inDims num.Dims,
	layers layer.Layers[data],
	device nnet.Device[data],
	optimizer Optimizer[data],
) *Sequential[data] {
	return &Sequential[data]{
		inDims:    inDims,
		Layers:    layers,
		device:    device,
		optimizer: optimizer,
	}
}

type Sequential[data any] struct {
	inDims num.Dims
	inputs data
	output data
	Layers layer.Layers[data]
	update []data
	device nnet.Device[data]

	optimizer  Optimizer[data]
	updateFunc func(iteration int)
}

func (s *Sequential[data]) Compile() data {
	s.inputs = s.device.NewData(s.inDims)
	s.output = s.Layers.Compile(s.device, s.inputs)

	s.update = append(s.update, s.Layers.ForUpdate()...)
	if s.optimizer != nil {
		s.updateFunc = s.optimizer(s.update)
	}

	return s.output
}

func (s *Sequential[data]) GetInput() data {
	return s.inputs
}

func (s *Sequential[data]) GetOutput() data {
	return s.output
}

func (s *Sequential[data]) GetTrainableParamsCount() int {
	var result int
	for _, node := range s.update {
		result += s.device.GetDataLength(node)
	}
	return result
}

func (s *Sequential[data]) Update(iteration int) {
	s.updateFunc(iteration)
}

func (s *Sequential[data]) LoadFromFile(filename string) error {
	t := time.Now()
	config, err := os.ReadFile(filename)
	if err != nil && errors.Is(err, os.ErrNotExist) {
		log.Println("trained config not found (skip)")
		return nil
	}
	if err != nil {
		return fmt.Errorf("read file failed: %w", err)
	}
	fmt.Println("read config success:", time.Since(t))

	t = time.Now()
	if err = json.Unmarshal(config, s); err != nil {
		return fmt.Errorf("unmarshal config failed: %w", err)
	}
	fmt.Println("unmarshal success:", time.Since(t))
	return nil
}

func (s *Sequential[data]) SaveToFile(filename string) error {
	t := time.Now()
	nnBytes, err := json.Marshal(s)
	if err != nil {
		return fmt.Errorf("marshal model config failed: %w", err)
	}

	fmt.Println("marshal success:", time.Since(t))

	t = time.Now()
	if err := os.WriteFile(filename, nnBytes, os.ModePerm); err != nil {
		return fmt.Errorf("write model config failed: %w", err)
	}
	fmt.Println("save success:", time.Since(t))
	return nil
}
