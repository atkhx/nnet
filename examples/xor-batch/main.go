package main

import (
	"fmt"
	"math/rand"

	"github.com/atkhx/nnet/block"
	"github.com/atkhx/nnet/layer"
	"github.com/atkhx/nnet/loss"
	"github.com/atkhx/nnet/model"
	"github.com/atkhx/nnet/num"
)

var (
	allInputs  = [][]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}}
	allTargets = [][]float64{{0}, {1}, {1}, {0}}

	batchSize  = 4
	inputsSize = 2
	outputSize = 1
	hiddenSize = 10

	learningRate   = 0.1
	learningEpochs = 100_000
	statisticsStep = 1000
)

func main() {
	seqModel := model.NewSequential(inputsSize, batchSize, layer.Layers{
		// FC Block 1
		block.NewSequentialBlock(layer.Layers{
			layer.NewFC(hiddenSize, num.SigmoidGain),
			layer.NewBias(),
			layer.NewSigmoid(),
		}),
		// FC Block 2
		layer.NewFC(outputSize, num.SigmoidGain),
		layer.NewBias(),
		layer.NewSigmoid(),
	})

	seqModel.Compile()

	lossAvg := 0.0
	output := make([]float64, outputSize*batchSize)

	batchInputs := make([]float64, 0, inputsSize*batchSize)
	batchTarget := make([]float64, 0, outputSize*batchSize)
	for i := range allInputs {
		batchInputs = append(batchInputs, allInputs[i]...)
		batchTarget = append(batchTarget, allTargets[i]...)
	}

	for i := 0; i < learningEpochs; i++ {
		rand.Shuffle(len(allInputs), func(i, j int) {
			allInputs[i], allInputs[j] = allInputs[j], allInputs[i]
			allTargets[i], allTargets[j] = allTargets[j], allTargets[i]
		})

		seqModel.Forward(batchInputs, output)

		lossAvg += loss.RegressionMean(batchSize, batchTarget, output)

		seqModel.Backward(batchTarget)
		seqModel.Update(learningRate)

		if i > 0 && i%statisticsStep == 0 {
			lossAvg /= float64(statisticsStep)
			fmt.Println("loss", lossAvg)
			lossAvg = 0
		}
	}
}
