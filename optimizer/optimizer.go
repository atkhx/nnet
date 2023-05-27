package optimizer

import (
	"math"

	"github.com/atkhx/nnet/num"
)

const (
	Ro  = 0.95
	Eps = 0.000001

	// todo check if it useful
	l1Decay = 0.0
	l2Decay = 0.0
)

type getGradDeltaFn func(k int, gradient float64) float64

func NewOptimizer(newGetGradDelta func(nodes num.Nodes) getGradDeltaFn) func(nodes num.Nodes) func() {
	return func(nodes num.Nodes) func() {
		getGradDelta := newGetGradDelta(nodes)
		return func() {
			offset := 0
			for _, node := range nodes {
				for j := 0; j < len(node.Data); j++ {
					l1grad := l1Decay
					if node.Data[j] <= 0 {
						l1grad = -l1grad
					}

					l2grad := l2Decay * node.Data[j]
					gradient := l2grad + l1grad + node.Grad[j]

					node.Data[j] += getGradDelta(offset+j, gradient)
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

func Adadelta(ro, eps float64) func(nodes num.Nodes) func() {
	return NewOptimizer(func(nodes num.Nodes) getGradDeltaFn {
		weightsCount := getWeightsCount(nodes)
		gsum := num.NewFloat64s(weightsCount)
		xsum := num.NewFloat64s(weightsCount)

		return func(k int, gradient float64) float64 {
			gsum[k] = ro*gsum[k] + (1-ro)*gradient*gradient
			value := -math.Sqrt((xsum[k]+eps)/(gsum[k]+eps)) * gradient
			xsum[k] = ro*xsum[k] + (1-ro)*value*value

			return value
		}
	})
}

func Adagrad(learning, eps float64) func(nodes num.Nodes) func() {
	return NewOptimizer(func(nodes num.Nodes) getGradDeltaFn {
		gsum := num.NewFloat64s(getWeightsCount(nodes))
		return func(k int, gradient float64) float64 {
			gsum[k] += gradient * gradient
			return -learning / math.Sqrt(gsum[k]+eps) * gradient
		}
	})
}

func VanilaSGD(learning float64) func(nodes num.Nodes) func() {
	return NewOptimizer(func(nodes num.Nodes) getGradDeltaFn {
		return func(k int, gradient float64) float64 {
			return -learning * gradient
		}
	})
}

func Nesterov(momentum, learning float64) func(nodes num.Nodes) func() {
	return NewOptimizer(func(nodes num.Nodes) getGradDeltaFn {
		gsum := num.NewFloat64s(getWeightsCount(nodes))
		return func(k int, gradient float64) float64 {
			value := gsum[k]
			gsum[k] = gsum[k]*momentum + learning*gradient
			return momentum*value - (1.0+momentum)*gsum[k]
		}
	})
}

func Momentum(momentum, learning float64) func(nodes num.Nodes) func() {
	return NewOptimizer(func(nodes num.Nodes) getGradDeltaFn {
		gsum := num.NewFloat64s(getWeightsCount(nodes))
		return func(k int, gradient float64) float64 {
			gsum[k] = gsum[k]*momentum - learning*gradient
			return gsum[k]
		}
	})
}
