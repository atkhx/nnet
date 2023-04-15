package main

import (
	"fmt"

	"github.com/atkhx/nnet/block"
	"github.com/atkhx/nnet/layer"
	"github.com/atkhx/nnet/loss"
	"github.com/atkhx/nnet/model"
)

var (
	allInputs  = [][]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}}
	allTargets = [][]float64{{0}, {1}, {1}, {0}}

	inputsSize = 2
	outputSize = 1
	hiddenSize = 10

	learningRate   = 0.01
	learningEpochs = 100_000
	statisticsStep = 1000
)

func main() {
	seqModel := model.NewSequential(inputsSize, []model.Layer{
		// FC Block 1
		block.NewSequentialBlock([]model.Layer{
			layer.NewFC(hiddenSize),
			layer.NewBias(),
			layer.NewSigmoid(),
		}),
		// FC Block 2
		block.NewSequentialBlock([]model.Layer{
			layer.NewFC(outputSize),
			layer.NewBias(),
			layer.NewSigmoid(),
		}),
	})

	seqModel.Compile()

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
