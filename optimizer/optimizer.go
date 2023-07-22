package optimizer

import (
	"math"

	"github.com/atkhx/nnet/num"
)

const (
	Ro  = 0.95
	Eps = 0.000001
)

type getGradDeltaFn func(iteration, k int, gradient float64) float64

func NewOptimizer(newGetGradDelta func(nodes num.Nodes) getGradDeltaFn) func(nodes num.Nodes) func(iteration int) {
	return func(nodes num.Nodes) func(iteration int) {
		getGradDelta := newGetGradDelta(nodes)
		return func(iteration int) {
			offset := 0
			for _, node := range nodes {
				for j := 0; j < len(node.Data); j++ {
					node.Data[j] += getGradDelta(iteration, offset+j, node.Grad[j])
				}
				offset += len(node.Data)
			}
		}
	}
}

func getWeightsCount(nodes num.Nodes) (weightsCount int) {
	for _, node := range nodes {
		weightsCount += len(node.Data)
	}
	return
}

func Adam(beta1, beta2, learningRate, eps float64) func(nodes num.Nodes) func(iteration int) {
	// beta1 - decay_rate1 или momentum
	// beta2 - decay_rate2 или adagrad_decay
	return NewOptimizer(func(nodes num.Nodes) getGradDeltaFn {
		weightsCount := getWeightsCount(nodes)

		m := num.NewFloat64s(weightsCount)
		v := num.NewFloat64s(weightsCount)

		beta1pow := num.NewFloat64s(5000)
		beta2pow := num.NewFloat64s(5000)

		for i := 0; i < 5000; i++ {
			if i == 0 {
				beta1pow[i] = beta1
				beta2pow[i] = beta2
			} else {
				beta1pow[i] = beta1pow[i-1] * beta1
				beta2pow[i] = beta2pow[i-1] * beta2
			}
		}

		for i := 0; i < 5000; i++ {
			beta1pow[i] = 1 / (1 - beta1pow[i])
			beta2pow[i] = 1 / (1 - beta2pow[i])
		}

		//pbeta1 := beta1
		//pbeta2 := beta2

		// m = vector add (
		return func(iteration, k int, gradient float64) float64 {
			m[k] = beta1*m[k] + (1-beta1)*gradient
			v[k] = beta2*v[k] + (1-beta2)*gradient*gradient

			//mHat := m[k] / (1 - pbeta1)
			//vHat := v[k] / (1 - pbeta2)

			//pbeta1 *= beta1
			//pbeta2 *= beta2

			//mHat := m[k] / (1 - math.Pow(beta1, float64(iteration)))
			//vHat := v[k] / (1 - math.Pow(beta2, float64(iteration)))

			//mHat := m[k] / (1 - beta1pow[iteration])
			//vHat := v[k] / (1 - beta2pow[iteration])

			mHat := m[k] * beta1pow[iteration]
			vHat := v[k] * beta2pow[iteration]

			return -learningRate * mHat / (math.Sqrt(vHat) + eps)
		}
	})
}

func Adadelta(ro, eps float64) func(nodes num.Nodes) func(iteration int) {
	return NewOptimizer(func(nodes num.Nodes) getGradDeltaFn {
		weightsCount := getWeightsCount(nodes)
		gsum := num.NewFloat64s(weightsCount)
		xsum := num.NewFloat64s(weightsCount)

		return func(_, k int, gradient float64) float64 {
			gsum[k] = ro*gsum[k] + (1-ro)*gradient*gradient
			value := -math.Sqrt((xsum[k]+eps)/(gsum[k]+eps)) * gradient
			xsum[k] = ro*xsum[k] + (1-ro)*value*value

			return value
		}
	})
}

func Adagrad(learning, eps float64) func(nodes num.Nodes) func(iteration int) {
	return NewOptimizer(func(nodes num.Nodes) getGradDeltaFn {
		gsum := num.NewFloat64s(getWeightsCount(nodes))
		return func(_, k int, gradient float64) float64 {
			gsum[k] += gradient * gradient
			return -learning / math.Sqrt(gsum[k]+eps) * gradient
		}
	})
}

func VanilaSGD(learning float64) func(nodes num.Nodes) func(iteration int) {
	return NewOptimizer(func(nodes num.Nodes) getGradDeltaFn {
		return func(_, k int, gradient float64) float64 {
			return -learning * gradient
		}
	})
}

func Nesterov(momentum, learning float64) func(nodes num.Nodes) func(iteration int) {
	return NewOptimizer(func(nodes num.Nodes) getGradDeltaFn {
		gsum := num.NewFloat64s(getWeightsCount(nodes))
		return func(_, k int, gradient float64) float64 {
			value := gsum[k]
			gsum[k] = gsum[k]*momentum + learning*gradient
			return momentum*value - (1.0+momentum)*gsum[k]
		}
	})
}

func Momentum(momentum, learning float64) func(nodes num.Nodes) func(iteration int) {
	return NewOptimizer(func(nodes num.Nodes) getGradDeltaFn {
		gsum := num.NewFloat64s(getWeightsCount(nodes))
		return func(_, k int, gradient float64) float64 {
			gsum[k] = gsum[k]*momentum - learning*gradient
			return gsum[k]
		}
	})
}
