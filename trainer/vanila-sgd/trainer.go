//go:generate mockgen -package=mocks -source=$GOFILE -destination=mocks/$GOFILE
package vanila_sgd

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/data"
)

type Net interface {
	Activate(inputs *data.Data) (output *data.Data)
	Backprop(deltas *data.Data) (gradient *data.Data)
	GetLayersCount() int
	GetLayer(index int) nnet.Layer
}

type Loss interface {
	GetDeltas(target, output *data.Data) (res *data.Data)
}

type TrainableLayer interface {
	nnet.Layer
	GetWeightsWithGradient() (w, g *data.Data)
	GetBiasesWithGradient() (w, g *data.Data)
}

func New(net Net, loss Loss, learningRate float64) *trainer {
	return &trainer{
		net:  net,
		loss: loss,

		learnRate: learningRate,
		output:    nil,
		deltas:    nil,
	}
}

type trainer struct {
	net  Net
	loss Loss

	learnRate float64

	output *data.Data
	deltas *data.Data
}

func (t *trainer) Activate(inputs, target *data.Data) *data.Data {
	t.output = t.net.Activate(inputs)
	t.deltas = t.loss.GetDeltas(target, t.output)
	t.net.Backprop(t.deltas)
	return t.output
}

func (t *trainer) UpdateWeights() {
	for i := 0; i < t.net.GetLayersCount(); i++ {
		layer, ok := t.net.GetLayer(i).(TrainableLayer)
		if ok {
			{
				w, g := layer.GetWeightsWithGradient()
				for j := 0; j < len(w.Data); j++ {
					w.Data[j] -= t.learnRate * g.Data[j]
				}
			}

			{
				w, g := layer.GetBiasesWithGradient()
				for j := 0; j < len(w.Data); j++ {
					w.Data[j] -= t.learnRate * g.Data[j]
				}
			}
		}
	}
}
