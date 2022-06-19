//go:generate mockgen -package=mocks -source=$GOFILE -destination=mocks/$GOFILE
package vanila_sgd

import (
	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/layer"
)

type Net interface {
	Activate(inputs *data.Data) (output *data.Data)
	Backprop(deltas *data.Data) (gradient *data.Data)
	GetLayersCount() int
	GetLayer(index int) layer.Layer
}

type Loss interface {
	GetDeltas(target, output *data.Data) (res *data.Data)
}

type TrainableLayer interface {
	layer.Layer
	GetWeightsWithGradient() (w, g *data.Data)
	GetBiasesWithGradient() (w, g *data.Data)
	ResetGradients()
}

func New(net Net, loss Loss, learnRate float64, batchSize int) *trainer {
	if batchSize < 1 {
		batchSize = 1
	}

	return &trainer{
		net:  net,
		loss: loss,

		batchSize: batchSize,
		learnRate: learnRate,
	}
}

type trainer struct {
	net  Net
	loss Loss

	batchSize  int
	batchIndex int
	learnRate  float64

	output *data.Data
	deltas *data.Data
}

func (t *trainer) Activate(inputs, target *data.Data) *data.Data {
	t.output = t.net.Activate(inputs).Copy()
	t.deltas = t.loss.GetDeltas(target, t.output)

	t.net.Backprop(t.deltas)
	t.batchIndex++

	return t.output
}

func (t *trainer) UpdateWeights() {
	if t.batchIndex != t.batchSize {
		return
	}

	t.batchIndex = 0
	batchRate := 1 / float64(t.batchSize)

	for i := 0; i < t.net.GetLayersCount(); i++ {
		iLayer, ok := t.net.GetLayer(i).(TrainableLayer)
		if ok {
			{
				w, g := iLayer.GetWeightsWithGradient()
				for j := 0; j < len(w.Data); j++ {
					w.Data[j] -= t.learnRate * g.Data[j] * batchRate
				}
			}

			{
				w, g := iLayer.GetBiasesWithGradient()
				for j := 0; j < len(w.Data); j++ {
					w.Data[j] -= t.learnRate * g.Data[j] * batchRate
				}
			}
			iLayer.ResetGradients()
		}
	}
}
