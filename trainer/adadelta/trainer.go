//go:generate mockgen -package=mocks -source=$GOFILE -destination=mocks/$GOFILE
package adadelta

import (
	"math"

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

func New(net Net, loss Loss, batchSize int) *trainer {
	if batchSize < 1 {
		batchSize = 1
	}

	return &trainer{
		net:  net,
		loss: loss,

		batchSize: batchSize,
	}
}

type trainer struct {
	net  Net
	loss Loss

	batchSize  int
	batchIndex int

	output *data.Data
	deltas *data.Data
	gsum   []*data.Data
	xsum   []*data.Data
}

const ro = 0.95
const eps = 0.000001

func (t *trainer) initGradients() {
	t.gsum = []*data.Data{}
	t.xsum = []*data.Data{}
	for i := 0; i < t.net.GetLayersCount(); i++ {
		if iLayer, ok := t.net.GetLayer(i).(TrainableLayer); ok {
			_, g := iLayer.GetWeightsWithGradient()
			t.gsum = append(t.gsum, g.CopyZero())
			t.xsum = append(t.xsum, g.CopyZero())

			_, g = iLayer.GetBiasesWithGradient()
			t.gsum = append(t.gsum, g.CopyZero())
			t.xsum = append(t.xsum, g.CopyZero())
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

const l1_decay = 0.001
const l2_decay = 0.001

func (t *trainer) UpdateWeights() {
	if len(t.gsum) == 0 {
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
					l1grad := l1_decay
					if w.Data[j] <= 0 {
						l1grad = -l1grad
					}

					l2grad := l2_decay * w.Data[j]
					gradient := (l2grad + l1grad + g.Data[j]) * batchRate

					t.gsum[k].Data[j] = ro*t.gsum[k].Data[j] + (1-ro)*gradient*gradient

					value := -math.Sqrt((t.xsum[k].Data[j]+eps)/(t.gsum[k].Data[j]+eps)) * gradient
					t.xsum[k].Data[j] = ro*t.xsum[k].Data[j] + (1-ro)*value*value
					w.Data[j] += value
				}
			}
			k++

			{
				w, g := iLayer.GetBiasesWithGradient()
				for j := 0; j < len(w.Data); j++ {
					l1grad := l1_decay
					if w.Data[j] <= 0 {
						l1grad = -l1grad
					}

					l2grad := l2_decay * w.Data[j]
					gradient := (l2grad + l1grad + g.Data[j]) * batchRate

					t.gsum[k].Data[j] = ro*t.gsum[k].Data[j] + (1-ro)*gradient*gradient

					value := -math.Sqrt((t.xsum[k].Data[j]+eps)/(t.gsum[k].Data[j]+eps)) * gradient
					t.xsum[k].Data[j] = ro*t.xsum[k].Data[j] + (1-ro)*value*value
					w.Data[j] += value
				}
			}

			k++
			iLayer.ResetGradients()
		}
	}
}
