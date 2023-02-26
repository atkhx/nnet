package main

import (
	"fmt"
	"math/rand"
	"strings"

	"github.com/atkhx/nnet/data"
)

func mRand(size int) []float64 {
	r := make([]float64, size)
	for i := range r {
		r[i] = rand.Float64()
	}
	return r
}

var (
	inputSize  = 2
	batchSize  = 4
	layer1Size = 2
	layer2Size = 1
)

func main() {
	rand.Seed(1)
	statChunk := 10
	epochs := 1_000
	lastIndex := epochs - 1

	// 	statChunk := 10
	//	epochs := 1000
	//	lastIndex := epochs - 1

	// loss [0.0034823765293007796]
	// loss [0.0033491507951614883]
	// loss [0.0032248854559231007]
	// ----------------------------------------
	// [0.004616808607349484 0.9446115284723823 0.9446173338970565 0.009142923502114918]
	// ----------------------------------------

	inputs := data.NewMatrix(inputSize, batchSize, []float64{
		0, 0,
		1, 0,
		0, 1,
		1, 1,
	})

	targets := data.NewMatrix(batchSize, 1, []float64{
		0, 1, 1, 0,
	})

	weights1 := data.NewMatrix(layer1Size, inputSize, mRand(layer1Size*inputSize))
	weights2 := data.NewMatrix(layer2Size, layer1Size, mRand(layer2Size*layer1Size))

	b1 := data.NewMatrix(layer1Size, 1, make([]float64, layer1Size))
	b2 := data.NewMatrix(layer2Size, 1, make([]float64, layer2Size))

	updateWeights := []*data.Matrix{weights1, weights2, b1, b2}

	for i := 0; i < epochs; i++ {
		outputs1 := inputs.MatrixMultiply(weights1)
		outputs1 = outputs1.AddRowVector(b1)

		output1Activated := outputs1.Tanh()
		//output1Activated := outputs1.Sigmoid()
		//output1Activated := outputs1.Relu()

		outputs2 := output1Activated.MatrixMultiply(weights2)
		outputs2 = outputs2.AddRowVector(b2)

		output2Activated := outputs2.Tanh().Transpose()
		//output2Activated := outputs2.Sigmoid()
		//output2Activated := outputs2.Relu()

		loss := output2Activated.Regression(targets)
		if i == 0 || i%statChunk == 0 {
			fmt.Println("loss", loss.Data)
		}

		if i == lastIndex {
			fmt.Println(strings.Repeat("-", 40))
			fmt.Println(output2Activated.Data)
			fmt.Println(strings.Repeat("-", 40))
		}

		loss.ResetGrad()
		loss.Backward()

		for _, weights := range updateWeights {
			for i, p := range weights.Grad {
				weights.Data[i] -= 0.1 * p
			}
		}
	}
}
