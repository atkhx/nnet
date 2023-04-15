package main

import (
	"fmt"

	"github.com/atkhx/nnet/layers"
	"github.com/atkhx/nnet/loss"
	"github.com/atkhx/nnet/model"
)

var (
	allInputs  = [][]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}}
	allTargets = [][]float64{{0}, {1}, {1}, {0}}

	learningRate   = 0.01
	learningEpochs = 100_000
	statisticsStep = 1000
)

func main() {
	seqModel := model.NewSequential(2)

	// FC Block 1
	seqModel.RegisterLayer(func(inputs, iGrads []float64) model.Layer {
		return layers.NewFC(10, inputs, iGrads)
	})

	seqModel.RegisterLayer(func(inputs, iGrads []float64) model.Layer {
		return layers.NewBias(inputs, iGrads)
	})

	seqModel.RegisterLayer(func(inputs, iGrads []float64) model.Layer {
		return layers.NewSigmoid(inputs, iGrads)
	})

	// FC Block 2
	seqModel.RegisterLayer(func(inputs, iGrads []float64) model.Layer {
		return layers.NewFC(1, inputs, iGrads)
	})

	seqModel.RegisterLayer(func(inputs, iGrads []float64) model.Layer {
		return layers.NewBias(inputs, iGrads)
	})

	seqModel.RegisterLayer(func(inputs, iGrads []float64) model.Layer {
		return layers.NewSigmoid(inputs, iGrads)
	})

	lossAvg := 0.0
	output := []float64{0}
	for i := 0; i < learningEpochs; i++ {
		for j := range allInputs {
			inputs := allInputs[j]
			target := allTargets[j]

			seqModel.Forward(inputs, output)
			lossAvg += loss.Regression(target, output)

			seqModel.Backward(target)
			seqModel.Update(learningRate)
		}

		if i > 0 && i%statisticsStep == 0 {
			lossAvg /= float64(len(allInputs) * statisticsStep)
			fmt.Println("loss", lossAvg)
			lossAvg = 0
		}
	}
}
