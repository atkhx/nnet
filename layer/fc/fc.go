package fc

import (
	"math"

	"github.com/atkhx/nnet/floats"

	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/executor"
)

func New(options ...Option) *FC {
	layer := &FC{}
	applyOptions(layer, defaults...)
	applyOptions(layer, options...)
	return layer
}

type FC struct {
	Weights *data.Data
	Biases  *data.Data

	Trainable bool

	iWidth, iHeight, iDepth, iVolume int
	OWidth, OHeight, ODepth, oVolume int

	inputs *data.Data
	output *data.Data

	gradInputs  *data.Data
	gradWeights *data.Data
	gradBiases  *data.Data

	gradInputsSeparated [][]float64
}

func (l *FC) InitDataSizes(w, h, d int) (int, int, int) {
	l.output = &data.Data{}
	l.output.Init3D(l.OWidth, l.OHeight, l.ODepth)

	l.iWidth, l.iHeight, l.iDepth = w, h, d

	l.iVolume = w * h * d
	l.oVolume = l.OWidth * l.OHeight * l.ODepth

	if l.Weights == nil {
		l.Weights = &data.Data{}
		l.Biases = &data.Data{}
	}

	if len(l.Weights.Data) == 0 {
		maxWeight := math.Sqrt(1.0 / float64(l.iVolume))

		l.Biases.Init3D(l.OWidth, l.OHeight, l.ODepth)
		l.Weights.Init4DRandom(l.iWidth, l.iHeight, l.iDepth, l.oVolume, 0, maxWeight)
	}

	l.gradInputs = &data.Data{}
	l.gradInputs.Init3D(l.iWidth, l.iHeight, l.iDepth)

	zcount := len(l.output.Data)
	l.gradInputsSeparated = make([][]float64, zcount)
	for i := 0; i < len(l.gradInputsSeparated); i++ {
		l.gradInputsSeparated[i] = make([]float64, l.iWidth*l.iHeight*l.iDepth)
	}

	l.gradBiases = &data.Data{}
	l.gradBiases.Init3D(l.OWidth, l.OHeight, l.ODepth)

	l.gradWeights = &data.Data{}
	l.gradWeights.Init4D(l.iWidth, l.iHeight, l.iDepth, l.OWidth*l.OHeight*l.ODepth)

	return l.OWidth, l.OHeight, l.ODepth
}

func (l *FC) Forward(inputs *data.Data) *data.Data {
	l.inputs = inputs
	executor.RunParallel(l.oVolume, func(i int) {
		l.output.Data[i] = l.Biases.Data[i] + l.inputs.Dot(l.Weights.Data[i*l.iVolume:])
	})
	return l.output
}

func (l *FC) ResetGradients() {
	l.gradWeights.FillZero()
	l.gradBiases.FillZero()
}

func (l *FC) Backward(deltas *data.Data) *data.Data {
	l.gradInputs.FillZero()
	l.gradBiases.Add(deltas.Data)

	executor.RunParallel(l.oVolume, func(i int) {
		ki := i * l.iVolume
		kj := ki + l.iVolume

		delta := deltas.Data[i]

		// weightsData := l.Weights.Data[ki:kj]
		// gradWeightsData := l.gradWeights.Data[ki:kj]
		// gradInputsData := l.gradInputsSeparated[i]

		// for j, iv := range weightsData {
		//	gradInputsData[j] = iv * delta
		// }
		floats.MultiplyTo(
			l.gradInputsSeparated[i],
			l.Weights.Data[ki:kj],
			delta,
		)

		// inputsData := l.inputs.Data
		// for j, iv := range inputsData {
		//	gradWeightsData[j] += iv * delta
		// }
		floats.MultiplyAndAddTo(
			l.gradWeights.Data[ki:kj],
			l.inputs.Data,
			delta,
		)
	})

	l.gradInputs.Add(l.gradInputsSeparated...)
	return l.gradInputs
}

func (l *FC) GetOutput() *data.Data {
	return l.output
}

func (l *FC) GetWeights() *data.Data {
	return l.Weights
}

func (l *FC) GetBiases() *data.Data {
	return l.Biases
}

func (l *FC) GetWeightsWithGradient() (*data.Data, *data.Data) {
	return l.Weights, l.gradWeights
}

func (l *FC) GetBiasesWithGradient() (*data.Data, *data.Data) {
	return l.Biases, l.gradBiases
}

func (l *FC) GetInputGradients() (g *data.Data) {
	return l.gradInputs
}

func (l *FC) GetWeightGradients() *data.Data {
	return l.gradWeights
}

func (l *FC) IsTrainable() bool {
	return l.Trainable
}
