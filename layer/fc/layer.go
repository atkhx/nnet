package fc

import (
	"github.com/atkhx/nnet/data"
	"math"
)

func New(options ...Option) *layer {
	layer := &layer{}
	defaults(layer)

	for _, opt := range options {
		opt(layer)
	}

	return layer
}

type layer struct {
	IWidth, IHeight, IDepth int
	OWidth, OHeight, ODepth int

	Weights *data.Data
	Biases  *data.Data

	inputs *data.Data
	output *data.Data

	gradWeights *data.Data
	gradBiases  *data.Data
	gradInputs  *data.Data

	iVolume int
}

func (l *layer) InitDataSizes(w, h, d int) (oW, oH, oD int) {
	l.output = &data.Data{}
	l.output.InitCube(l.OWidth, l.OHeight, l.ODepth)

	l.IWidth, l.IHeight, l.IDepth = w, h, d
	l.iVolume = w * h * d

	if l.Weights == nil {
		l.Weights = &data.Data{}
		l.Biases = &data.Data{}
	}

	if len(l.Weights.Data) == 0 {
		maxWeight := math.Sqrt(1.0 / float64(l.IWidth*l.IHeight*l.IDepth))

		l.Biases.InitCube(l.OWidth, l.OHeight, l.ODepth)
		l.Weights.InitHiperCubeRandom(l.IWidth, l.IHeight, l.IDepth, l.OWidth*l.OHeight*l.ODepth, 0, maxWeight)
	}

	l.gradInputs = &data.Data{}
	l.gradInputs.InitCube(l.IWidth, l.IHeight, l.IDepth)

	l.gradBiases = &data.Data{}
	l.gradBiases.InitCube(l.OWidth, l.OHeight, l.ODepth)

	l.gradWeights = &data.Data{}
	l.gradWeights.InitHiperCube(l.IWidth, l.IHeight, l.IDepth, l.OWidth*l.OHeight*l.ODepth)

	return l.OWidth, l.OHeight, l.ODepth
}

func (l *layer) Activate(inputs *data.Data) *data.Data {
	l.inputs = inputs

	for i := 0; i < len(l.output.Data); i++ {
		k := i * l.iVolume
		o := 0.0

		for j := 0; j < len(l.inputs.Data); j++ {
			o += l.Weights.Data[k+j] * l.inputs.Data[j]
		}

		l.output.Data[i] = o + l.Biases.Data[i]
	}

	return l.output
}

func (l *layer) Backprop(deltas *data.Data) *data.Data {
	l.gradInputs.Reset()
	l.gradWeights.Reset()

	l.gradBiases = deltas.Copy()

	for i := 0; i < len(l.output.Data); i++ {
		k := i * l.iVolume
		d := deltas.Data[i]

		for j := 0; j < len(l.inputs.Data); j++ {
			l.gradInputs.Data[j] += l.Weights.Data[k+j] * d
			l.gradWeights.Data[k+j] += l.inputs.Data[j] * d
		}
	}
	return l.gradInputs
}

func (l *layer) GetOutput() *data.Data {
	return l.output
}

func (l *layer) GetWeights() *data.Data {
	return l.Weights
}

func (l *layer) GetBiases() *data.Data {
	return l.Biases
}

func (l *layer) GetWeightsWithGradient() (*data.Data, *data.Data) {
	return l.Weights, l.gradWeights
}

func (l *layer) GetBiasesWithGradient() (*data.Data, *data.Data) {
	return l.Biases, l.gradBiases
}

func (l *layer) GetInputGradients() (g *data.Data) {
	return l.gradInputs
}
