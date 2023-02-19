//go:generate mockgen -package=mocks -source=$GOFILE -destination=mocks/$GOFILE
package trainer

import (
	"github.com/atkhx/nnet/data"
)

func New(net Net, loss LossFunc, opts ...Option) Trainer {
	res := &trainer{net: net, lossFunc: loss}
	applyOptions(res, defaults...)
	applyOptions(res, opts...)

	res.batchRate = 1 / float64(res.batchSize)
	res.method.Init(res.getWeightsCount())
	return res
}

type trainer struct {
	Trainer

	net Net

	lossFunc  LossFunc
	lossValue float64

	batchSize  int
	batchIndex int
	batchRate  float64

	l1Decay float64
	l2Decay float64

	method Method
}

func (t *trainer) GetLossFunc() LossFunc {
	return t.lossFunc
}

func (t *trainer) GetLossValue() float64 {
	return t.lossValue
}

func (t *trainer) getWeightsCount() (weightsCount int) {
	for i := 0; i < t.net.GetLayersCount(); i++ {
		if iLayer, ok := t.net.GetLayer(i).(TrainableLayer); ok && iLayer.IsTrainable() {
			_, g := iLayer.GetWeightsWithGradient()
			weightsCount += len(g.Data)

			_, g = iLayer.GetBiasesWithGradient()
			weightsCount += len(g.Data)
		}
	}
	return
}

func (t *trainer) Forward(inputs, target *data.Data) *data.Data {
	// we copy output for return, because
	// firstly it's refers to inner layer property output
	// and could be changed on next Forward (or Backward, who knows)

	output := t.net.Forward(inputs).Copy()
	deltas := t.lossFunc.GetDeltas(target, output)

	t.lossValue = t.lossFunc.GetError(target.Data, output.Data)

	t.net.Backward(deltas)
	t.batchIndex++
	return output
}

func (t *trainer) ForwardFn(forwardFn func()) {
	forwardFn()
	t.batchIndex++
}

func (t *trainer) UpdateWeights() {
	if t.batchIndex != t.batchSize {
		return
	}

	t.batchIndex = 0

	k := 0
	for i := 0; i < t.net.GetLayersCount(); i++ {
		iLayer, ok := t.net.GetLayer(i).(TrainableLayer)
		if ok && iLayer.IsTrainable() {
			w, g := iLayer.GetWeightsWithGradient()
			t.updateWeights(k, w, g)
			k += len(g.Data)

			w, g = iLayer.GetBiasesWithGradient()
			t.updateWeights(k, w, g)
			k += len(g.Data)

			iLayer.ResetGradients()
		}
	}
}

func (t *trainer) updateWeights(offset int, w, g *data.Data) {
	for j := 0; j < len(w.Data); j++ {
		l1grad := t.l1Decay
		if w.Data[j] <= 0 {
			l1grad = -l1grad
		}

		l2grad := t.l2Decay * w.Data[j]
		gradient := (l2grad + l1grad + g.Data[j]) * t.batchRate

		w.Data[j] += t.method.GetDelta(offset+j, gradient)
	}
}
