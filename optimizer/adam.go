package optimizer

import (
	"math"

	"github.com/atkhx/nnet/num"
)

func NewOptimizerAdam(iterations int, beta1, beta2, learningRate, eps float64) func(nodes num.Nodes) func(iteration int) {
	iterations++
	return func(nodes num.Nodes) func(iteration int) {

		weightsCount := getWeightsCount(nodes)

		m := num.NewFloat64s(weightsCount)
		v := num.NewFloat64s(weightsCount)

		beta1pow := num.NewFloat64s(iterations)
		beta2pow := num.NewFloat64s(iterations)

		for i := 0; i < iterations; i++ {
			if i == 0 {
				beta1pow[i] = beta1
				beta2pow[i] = beta2
			} else {
				beta1pow[i] = beta1pow[i-1] * beta1
				beta2pow[i] = beta2pow[i-1] * beta2
			}
		}

		for i := 0; i < iterations; i++ {
			beta1pow[i] = 1 / (1 - beta1pow[i])
			beta2pow[i] = 1 / (1 - beta2pow[i])
		}

		beta1o := 1 - beta1
		beta2o := 1 - beta2

		return func(iteration int) {
			offset := 0
			for _, node := range nodes {
				m := m[offset : offset+len(node.Data)]
				v := v[offset : offset+len(node.Data)]
				offset += len(node.Data)

				m.MulScalar(beta1)
				m.AddWeighted(node.Grad, beta1o)

				node.Grad.Pow2()
				v.MulScalar(beta2)
				v.AddWeighted(node.Grad, beta2o)

				for j := range node.Data {
					node.Data[j] -= learningRate * m[j] * beta1pow[iteration] / (math.Sqrt(v[j]*beta2pow[iteration]) + eps)
				}
			}
		}
	}
}
