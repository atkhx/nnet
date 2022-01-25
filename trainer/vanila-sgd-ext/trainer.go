//go:generate mockgen -package=mocks -source=$GOFILE -destination=mocks/$GOFILE
package vanila_sgd_ext

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

func New(net Net, loss Loss, learning, momentum, weightDecay float64) *trainer {
	return &trainer{
		net:         net,
		loss:        loss,
		learnRate:   learning,
		momentum:    momentum,
		weightDecay: weightDecay,
	}
}

type trainer struct {
	net  Net
	loss Loss

	learnRate   float64
	momentum    float64
	weightDecay float64

	output    *data.Data
	deltas    *data.Data
	gradients []*data.Data
}

func (t *trainer) initGradients() {
	t.gradients = []*data.Data{}
	for i := 0; i < t.net.GetLayersCount(); i++ {
		if layer, ok := t.net.GetLayer(i).(TrainableLayer); ok {
			_, g := layer.GetWeightsWithGradient()
			t.gradients = append(t.gradients, g.CopyZero())

			_, g = layer.GetBiasesWithGradient()
			t.gradients = append(t.gradients, g.CopyZero())
		}
	}
}

func (t *trainer) Activate(inputs, target *data.Data) *data.Data {
	t.output = t.net.Activate(inputs).Copy()
	t.deltas = t.loss.GetDeltas(target, t.output)

	t.net.Backprop(t.deltas)

	return t.output
}

func (t *trainer) UpdateWeights() {
	if len(t.gradients) == 0 {
		t.initGradients()
	}

	k := 0
	for i := 0; i < t.net.GetLayersCount(); i++ {
		layer, ok := t.net.GetLayer(i).(TrainableLayer)
		if ok {
			{
				w, g := layer.GetWeightsWithGradient()
				for j := 0; j < len(w.Data); j++ {
					value := t.gradients[k].Data[j]*t.momentum + t.learnRate*g.Data[j] + t.weightDecay*w.Data[j]

					w.Data[j] -= value
					t.gradients[k].Data[j] = value
				}
			}
			k++

			{
				w, g := layer.GetBiasesWithGradient()
				for j := 0; j < len(w.Data); j++ {
					value := t.gradients[k].Data[j]*t.momentum + t.learnRate*g.Data[j] + t.weightDecay*w.Data[j]

					w.Data[j] -= value
					t.gradients[k].Data[j] = value
				}
			}

			k++
		}
	}
}
