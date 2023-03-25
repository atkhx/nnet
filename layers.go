package nnet

import "math/rand"

func NewFullyConnectedLayer(inputSize, batchSize, nCount int) func(inputs, output []float64) {
	weights := make([]float64, inputSize*nCount)
	biases := make([]float64, nCount)

	for i := range weights {
		weights[i] = rand.NormFloat64()
	}

	return func(inputs, output []float64) {
		for b := 0; b < batchSize; b++ {
			inputs := inputs[b*inputSize : b*inputSize+inputSize]
			output := output[b*nCount : b*nCount+nCount]

			for i, v := range output {
				weights := weights[i*inputSize : i*inputSize+inputSize]
				for j, w := range weights[i*inputSize : i*inputSize+inputSize] {
					v += w * inputs[j]
				}
				v += biases[i]

				output[i] = v + biases[i]
			}
		}
	}
}
