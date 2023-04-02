package main

import (
	"fmt"
	"math/rand"

	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/layer/activation"
	"github.com/atkhx/nnet/layer/fc"
	"github.com/atkhx/nnet/net"
	"github.com/atkhx/nnet/trainer"
)

var (
	inputSize = 2
)

func main() {
	rand.Seed(123)

	statChunk := 1000
	epochs := 10_000

	inputs := []*data.Data{
		data.WrapData(inputSize, 1, 1, []float64{0, 0}),
		data.WrapData(inputSize, 1, 1, []float64{1, 0}),
		data.WrapData(inputSize, 1, 1, []float64{0, 1}),
		data.WrapData(inputSize, 1, 1, []float64{1, 1}),
	}

	targets := []*data.Data{
		data.WrapData(1, 1, 1, []float64{0}),
		data.WrapData(1, 1, 1, []float64{1}),
		data.WrapData(1, 1, 1, []float64{1}),
		data.WrapData(1, 1, 1, []float64{0}),
	}

	nn := net.New(net.Layers{
		fc.New(
			fc.WithInputSize(inputSize),
			fc.WithLayerSize(3),
			fc.WithBiases(true),
		),
		activation.NewSigmoid(),
		fc.New(
			fc.WithInputSize(3),
			fc.WithLayerSize(1),
			fc.WithBiases(true),
		),
		activation.NewSigmoid(),
	})

	nnTrainer := trainer.New(
		nn,
	)

	for e := 0; e < epochs; e++ {
		for i := range inputs {
			loss := nnTrainer.Forward(inputs[i], func(output *data.Data) *data.Data {
				return output.Regression(targets[i])
			})

			if e == 0 || e%statChunk == 0 {
				fmt.Println("loss:", loss.Data)
			}
		}
	}

	fmt.Println("output")
	for i := range inputs {
		fmt.Print(nn.Forward(inputs[i]).Data)
	}
}
