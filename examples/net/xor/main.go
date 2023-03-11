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
	batchSize = 4
)

func main() {
	rand.Seed(123)

	statChunk := 1000
	epochs := 10000

	inputs := data.NewData(inputSize, batchSize, 1, []float64{
		0, 0,
		1, 0,
		0, 1,
		1, 1,
	})

	targets := data.NewData(1, batchSize, 1, []float64{
		0,
		1,
		1,
		0,
	})

	nn := net.New(net.Layers{
		fc.New(
			fc.WithInputSize(inputSize),
			fc.WithLayerSize(3),
		),
		activation.NewSigmoid(),
		fc.New(
			fc.WithInputSize(3),
			fc.WithLayerSize(1),
		),
		activation.NewSigmoid(),
	})

	nnTrainer := trainer.New(
		nn,
		//trainer.WithMethod(methods.Adadelta(trainer.Ro, trainer.Eps)),
		//trainer.WithMethod(methods.Adagard(0.1, trainer.Eps)),
		//trainer.WithMethod(methods.Nesterov(0.1, 0.7)),
		//trainer.WithMethod(methods.Momentum(0.1, 0.7)),
		//trainer.WithMethod(methods.VanilaSGD(0.1)),
	)

	for e := 0; e < epochs; e++ {
		loss := nnTrainer.Forward(inputs, func(output *data.Data) *data.Data {
			return output.Regression(targets)
		})

		if e == 0 || e%statChunk == 0 {
			fmt.Println("loss:", loss.Data)
		}
	}

	fmt.Println("output", nn.Forward(inputs).Data)
}
