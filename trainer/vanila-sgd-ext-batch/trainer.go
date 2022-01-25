//go:generate mockgen -package=mocks -source=$GOFILE -destination=mocks/$GOFILE
package vanila_sgd_ext_batch

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
}

func New(net Net, loss Loss, batchSize int, learning, momentum, weightDecay float64) *trainer {
	return &trainer{
		net:  net,
		loss: loss,

		learnRate:   learning,
		momentum:    momentum,
		weightDecay: weightDecay,
		batchSize:   batchSize,
	}
}

type trainer struct {
	net  Net
	loss Loss

	learnRate   float64
	momentum    float64
	weightDecay float64
	batchSize   int
	batchIndex  int

	output    *data.Data
	deltas    *data.Data
	gradients []*data.Data
	sumGrads  []*data.Data
}

func (t *trainer) initGradients() {
	t.gradients = []*data.Data{}
	t.sumGrads = []*data.Data{}
	for i := 0; i < t.net.GetLayersCount(); i++ {
		if layer, ok := t.net.GetLayer(i).(TrainableLayer); ok {
			_, g := layer.GetWeightsWithGradient()
			t.gradients = append(t.gradients, g.CopyZero())
			t.sumGrads = append(t.sumGrads, g.CopyZero())

			_, g = layer.GetBiasesWithGradient()
			t.gradients = append(t.gradients, g.CopyZero())
			t.sumGrads = append(t.sumGrads, g.CopyZero())
		}
	}
}

func (t *trainer) Activate(inputs, target *data.Data) *data.Data {
	t.output = t.net.Activate(inputs)
	t.deltas = t.loss.GetDeltas(target, t.output)

	t.net.Backprop(t.deltas)

	if len(t.gradients) == 0 {
		t.initGradients()
	}

	if t.batchSize < 1 {
		t.batchSize = 1
	}

	t.batchIndex++

	k := 0
	for i := 0; i < t.net.GetLayersCount(); i++ {
		layer, ok := t.net.GetLayer(i).(TrainableLayer)
		if ok {
			{
				_, g := layer.GetWeightsWithGradient()
				for j := 0; j < len(g.Data); j++ {
					t.sumGrads[k].Data[j] += g.Data[j]
				}
			}
			k++

			{
				_, g := layer.GetBiasesWithGradient()
				for j := 0; j < len(g.Data); j++ {
					t.sumGrads[k].Data[j] += g.Data[j]
				}
			}
			k++
		}
	}

	return t.output
}

func (t *trainer) UpdateWeights() {
	if t.batchIndex == t.batchSize {
		t.batchIndex = 0
	} else {
		return
	}

	batchRate := 1 / float64(t.batchSize)
	k := 0
	for i := 0; i < t.net.GetLayersCount(); i++ {
		layer, ok := t.net.GetLayer(i).(TrainableLayer)
		if ok {
			{
				w, _ := layer.GetWeightsWithGradient()
				for j := 0; j < len(w.Data); j++ {
					value := t.gradients[k].Data[j]*t.momentum + t.sumGrads[k].Data[j]*batchRate*t.learnRate

					w.Data[j] -= value
					t.gradients[k].Data[j] = value
					t.sumGrads[k].Data[j] = 0
				}
			}
			k++

			{
				w, _ := layer.GetBiasesWithGradient()
				for j := 0; j < len(w.Data); j++ {
					value := t.gradients[k].Data[j]*t.momentum + t.sumGrads[k].Data[j]*batchRate*t.learnRate

					w.Data[j] -= value
					t.gradients[k].Data[j] = value
					t.sumGrads[k].Data[j] = 0
				}
			}

			k++
		}
	}
}
