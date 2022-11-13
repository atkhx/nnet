package fc

import (
	"math"

	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/executor"
)

func New(options ...Option) *Layer {
	layer := &Layer{}
	applyOptions(layer, defaults...)
	applyOptions(layer, options...)

	return layer
}

type Layer struct {
	// begin storable Layer config
	Weights *data.Data
	Biases  *data.Data

	Trainable bool
	// end storable Layer config

	iWidth, iHeight, iDepth, iVolume int
	oWidth, oHeight, oDepth, oVolume int

	inputs *data.Data
	output *data.Data

	gradInputs  *data.Data
	gradWeights *data.Data
	gradBiases  *data.Data

	gradInputsSeparated [][]float64
}

func (l *Layer) InitDataSizes(w, h, d int) (int, int, int) {
	l.output = &data.Data{}
	l.output.InitCube(l.oWidth, l.oHeight, l.oDepth)

	l.iWidth, l.iHeight, l.iDepth = w, h, d

	l.iVolume = w * h * d
	l.oVolume = l.oWidth * l.oHeight * l.oDepth

	if l.Weights == nil {
		l.Weights = &data.Data{}
		l.Biases = &data.Data{}
	}

	if len(l.Weights.Data) == 0 {
		maxWeight := math.Sqrt(1.0 / float64(l.iVolume))

		l.Biases.InitCube(l.oWidth, l.oHeight, l.oDepth)
		l.Weights.InitHiperCubeRandom(l.iWidth, l.iHeight, l.iDepth, l.oWidth*l.oHeight*l.oDepth, 0, maxWeight)
	}

	l.gradInputs = &data.Data{}
	l.gradInputs.InitCube(l.iWidth, l.iHeight, l.iDepth)

	zcount := len(l.output.Data)
	l.gradInputsSeparated = make([][]float64, zcount)
	for i := 0; i < len(l.gradInputsSeparated); i++ {
		l.gradInputsSeparated[i] = make([]float64, l.iWidth*l.iHeight*l.iDepth)
	}

	l.gradBiases = &data.Data{}
	l.gradBiases.InitCube(l.oWidth, l.oHeight, l.oDepth)

	l.gradWeights = &data.Data{}
	l.gradWeights.InitHiperCube(l.iWidth, l.iHeight, l.iDepth, l.oWidth*l.oHeight*l.oDepth)

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *Layer) Activate(inputs *data.Data) *data.Data {
	l.inputs = inputs
	executor.RunParallel(l.oVolume, func(i int) {
		l.output.Data[i] = l.Biases.Data[i] + l.inputs.Dot(l.Weights.Data[i*l.iVolume:])
	})
	return l.output
}

func (l *Layer) ResetGradients() {
	l.gradWeights.FillZero()
	l.gradBiases.FillZero()
}

func (l *Layer) Backprop(deltas *data.Data) *data.Data {
	l.gradInputs.FillZero()
	l.gradBiases.Add(deltas.Data)

	executor.RunParallel(l.oVolume, func(i int) {
		ki := i * l.iVolume
		kj := ki + l.iVolume

		delta := deltas.Data[i]

		// 2 cycles faster
		weightsData := l.Weights.Data[ki:kj]
		gradWeightsData := l.gradWeights.Data[ki:kj]
		gradInputsData := l.gradInputsSeparated[i]
		for j, iv := range weightsData {
			gradInputsData[j] = iv * delta
		}

		inputsData := l.inputs.Data
		for j, iv := range inputsData {
			gradWeightsData[j] += iv * delta
		}
	})

	l.gradInputs.Add(l.gradInputsSeparated...)
	return l.gradInputs
}

func (l *Layer) GetOutput() *data.Data {
	return l.output
}

func (l *Layer) GetWeights() *data.Data {
	return l.Weights
}

func (l *Layer) GetBiases() *data.Data {
	return l.Biases
}

func (l *Layer) GetWeightsWithGradient() (*data.Data, *data.Data) {
	return l.Weights, l.gradWeights
}

func (l *Layer) GetBiasesWithGradient() (*data.Data, *data.Data) {
	return l.Biases, l.gradBiases
}

func (l *Layer) GetInputGradients() (g *data.Data) {
	return l.gradInputs
}

func (l *Layer) GetWeightGradients() *data.Data {
	return l.gradWeights
}

func (l *Layer) IsTrainable() bool {
	return l.Trainable
}
