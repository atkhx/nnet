//go:generate mockgen -package=mocks -source=$GOFILE -destination=mocks/$GOFILE
package trainer

import (
	"github.com/atkhx/nnet/data"
)

func New(net Net, loss Loss, method Method, batchSize int) *trainer {
	if batchSize < 1 {
		batchSize = 1
	}

	res := &trainer{
		net:    net,
		loss:   loss,
		method: method,

		batchSize: batchSize,
	}

	res.method.Init(res.getWeightsCount())
	return res
}

type trainer struct {
	net  Net
	loss Loss

	batchSize  int
	batchIndex int

	method Method
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

func (t *trainer) Activate(inputs, target *data.Data) *data.Data {
	output := t.net.Activate(inputs).Copy()
	deltas := t.loss.GetDeltas(target, output)

	t.net.Backprop(deltas)
	t.batchIndex++

	return output
}

func (t *trainer) UpdateWeights() {
	if t.batchIndex != t.batchSize {
		return
	}

	t.batchIndex = 0
	batchRate := 1 / float64(t.batchSize)

	k := 0
	for i := 0; i < t.net.GetLayersCount(); i++ {
		iLayer, ok := t.net.GetLayer(i).(TrainableLayer)
		if ok && iLayer.IsTrainable() {
			w, g := iLayer.GetWeightsWithGradient()
			t.updateWeights(batchRate, k, w, g)
			k += len(g.Data)

			w, g = iLayer.GetBiasesWithGradient()
			t.updateWeights(batchRate, k, w, g)
			k += len(g.Data)

			iLayer.ResetGradients()
		}
	}
}

func (t *trainer) updateWeights(batchRate float64, offset int, w, g *data.Data) {
	for j := 0; j < len(w.Data); j++ {
		l1grad := l1Decay
		if w.Data[j] <= 0 {
			l1grad = -l1grad
		}

		l2grad := l2Decay * w.Data[j]
		gradient := (l2grad + l1grad + g.Data[j]) * batchRate

		w.Data[j] += t.method.GetDelta(offset+j, gradient)
	}
}
