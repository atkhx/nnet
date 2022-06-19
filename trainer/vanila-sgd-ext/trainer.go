//go:generate mockgen -package=mocks -source=$GOFILE -destination=mocks/$GOFILE
package vanila_sgd_ext

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

func New(net Net, loss Loss, learning, momentum, weightDecay float64, batchSize int) *trainer {
	if batchSize < 1 {
		batchSize = 1
	}

	return &trainer{
		net:  net,
		loss: loss,

		batchSize:   batchSize,
		learnRate:   learning,
		momentum:    momentum,
		weightDecay: weightDecay,
	}
}

type trainer struct {
	net  Net
	loss Loss

	batchSize   int
	batchIndex  int
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
		if iLayer, ok := t.net.GetLayer(i).(TrainableLayer); ok {
			_, g := iLayer.GetWeightsWithGradient()
			t.gradients = append(t.gradients, g.CopyZero())

			_, g = iLayer.GetBiasesWithGradient()
			t.gradients = append(t.gradients, g.CopyZero())
		}
	}
}

func (t *trainer) Activate(inputs, target *data.Data) *data.Data {
	t.output = t.net.Activate(inputs).Copy()
	t.deltas = t.loss.GetDeltas(target, t.output)

	t.net.Backprop(t.deltas)
	t.batchIndex++

	return t.output
}

func (t *trainer) UpdateWeights() {
	if len(t.gradients) == 0 {
		t.initGradients()
	}

	if t.batchIndex != t.batchSize {
		return
	}

	t.batchIndex = 0
	batchRate := 1 / float64(t.batchSize)

	k := 0
	for i := 0; i < t.net.GetLayersCount(); i++ {
		iLayer, ok := t.net.GetLayer(i).(TrainableLayer)
		if ok {
			{
				w, g := iLayer.GetWeightsWithGradient()
				for j := 0; j < len(w.Data); j++ {
					value := t.gradients[k].Data[j]*t.momentum - (t.learnRate*g.Data[j]*batchRate + t.weightDecay*w.Data[j])

					t.gradients[k].Data[j] = value
					w.Data[j] += value
				}
			}
			k++

			{
				w, g := iLayer.GetBiasesWithGradient()
				for j := 0; j < len(w.Data); j++ {
					value := t.gradients[k].Data[j]*t.momentum - (t.learnRate*g.Data[j]*batchRate + t.weightDecay*w.Data[j])

					t.gradients[k].Data[j] = value
					w.Data[j] += value
				}
			}
			k++
			iLayer.ResetGradients()
		}
	}
}
