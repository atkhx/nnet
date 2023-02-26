//go:generate mockgen -package=mocks -source=$GOFILE -destination=mocks/$GOFILE
package trainer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/data"
)

func New(net Net, opts ...Option) Trainer {
	res := &trainer{net: net}
	applyOptions(res, defaults...)
	applyOptions(res, opts...)

	res.method.Init(res.getWeightsCount())
	return res
}

type trainer struct {
	Trainer

	net Net

	l1Decay float64
	l2Decay float64

	method Method
}

// todo rename to getParamsCount
func (t *trainer) getWeightsCount() (weightsCount int) {
	for i := 0; i < t.net.GetLayersCount(); i++ {
		l := t.net.GetLayer(i)

		if withWeights, ok := l.(nnet.WithWeights); ok {
			if len(withWeights.GetWeights().Data) == 0 {
				panic("weights len is zero")
			}
			weightsCount += len(withWeights.GetWeights().Data)
		}

		if withBiases, ok := l.(nnet.WithBiases); ok && withBiases.HasBiases() {
			if len(withBiases.GetBiases().Data) == 0 {
				panic("biases len is zero")
			}
			weightsCount += len(withBiases.GetBiases().Data)
		}
	}
	return
}

func (t *trainer) Forward(inputs *data.Matrix, getLoss func(output *data.Matrix) *data.Matrix) *data.Matrix {
	output := t.net.Forward(inputs)

	loss := getLoss(output)
	loss.ResetGrad()
	loss.Backward()

	t.updateWeights()
	return loss
}

func (t *trainer) updateWeights() {
	k := 0
	for i := 0; i < t.net.GetLayersCount(); i++ {
		l := t.net.GetLayer(i)

		if withWeights, ok := l.(nnet.WithWeights); ok {
			weights := withWeights.GetWeights()
			t.updateWeightsWithUsingMethod(k, weights)

			k += len(weights.Data)
		}

		if withBiases, ok := l.(nnet.WithBiases); ok && withBiases.HasBiases() {
			biases := withBiases.GetBiases()
			t.updateWeightsWithUsingMethod(k, biases)

			k += len(biases.Data)
		}
	}
}

func (t *trainer) updateWeightsWithUsingMethod(offset int, w *data.Matrix) {
	for j := 0; j < len(w.Data); j++ {
		l1grad := t.l1Decay
		if w.Data[j] <= 0 {
			l1grad = -l1grad
		}

		l2grad := t.l2Decay * w.Data[j]
		gradient := l2grad + l1grad + w.Grad[j]

		w.Data[j] += t.method.GetDelta(offset+j, gradient)
	}
}
