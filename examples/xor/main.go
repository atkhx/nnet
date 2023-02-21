package main

import (
	"fmt"
	"math/rand"

	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/layer/activation"
	"github.com/atkhx/nnet/layer/fc"
	"github.com/atkhx/nnet/loss"
	"github.com/atkhx/nnet/net"
	"github.com/atkhx/nnet/trainer"
)

type Sample struct {
	Input  *data.Data
	Target *data.Data
}

func main() {
	rand.Seed(123)

	nn := net.New(2, 1, 1, net.Layers{
		fc.New(fc.OutputSizes(3, 3, 3)),
		activation.NewSigmoid(),
		fc.New(fc.OutputSizes(1, 1, 1)),
		activation.NewSigmoid(),
	})

	nn.Init()

	netTrainer := trainer.New(nn)

	samples := []Sample{
		{
			Input:  data.NewVectorFloats(0, 0),
			Target: data.NewVectorFloats(0),
		},
		{
			Input:  data.NewVectorFloats(1, 0),
			Target: data.NewVectorFloats(1),
		},
		{
			Input:  data.NewVectorFloats(0, 1),
			Target: data.NewVectorFloats(1),
		},
		{
			Input:  data.NewVectorFloats(1, 1),
			Target: data.NewVectorFloats(0),
		},
	}

	for e := 0; e < 10000; e++ {
		var avgLoss float64
		for _, sample := range samples {
			_, lossObject := netTrainer.Forward(sample.Input, loss.NewRegressionLossFunc(sample.Target))
			netTrainer.UpdateWeights()

			avgLoss += lossObject.GetError()
		}
		fmt.Println("avgLoss:", avgLoss/float64(len(samples)))
	}

	for _, sample := range samples {
		output := nn.Forward(sample.Input.Copy())
		fmt.Println(sample.Input.Data, "=>", sample.Target.Data, output.Data)
	}
}
