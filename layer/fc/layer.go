package fc

import (
	"math"
	"sync"

	"github.com/atkhx/nnet/data"
)

func New(options ...Option) *layer {
	layer := &layer{}
	applyOptions(layer, defaults...)
	applyOptions(layer, options...)

	return layer
}

type layer struct {
	// begin storable layer config
	Weights *data.Data
	Biases  *data.Data

	Trainable bool
	// end storable layer config

	iWidth, iHeight, iDepth int
	oWidth, oHeight, oDepth int

	inputs *data.Data
	output *data.Data
	deltas *data.Data

	gradInputs  *data.Data
	gradWeights *data.Data
	gradBiases  *data.Data

	iVolume int
	threads int

	activateInChan chan int
	backpropInChan chan int

	wg sync.WaitGroup
}

func (l *layer) InitDataSizes(w, h, d int) (oW, oH, oD int) {
	l.output = &data.Data{}
	l.output.InitCube(l.oWidth, l.oHeight, l.oDepth)

	l.iWidth, l.iHeight, l.iDepth = w, h, d
	l.iVolume = w * h * d

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

	l.gradBiases = &data.Data{}
	l.gradBiases.InitCube(l.oWidth, l.oHeight, l.oDepth)

	l.gradWeights = &data.Data{}
	l.gradWeights.InitHiperCube(l.iWidth, l.iHeight, l.iDepth, l.oWidth*l.oHeight*l.oDepth)

	if l.threads == 0 {
		l.threads = len(l.output.Data)
	}

	l.activateInChan = make(chan int, l.threads)
	l.backpropInChan = make(chan int, l.threads)

	for i := 0; i < l.threads; i++ {
		go func() {
			for {
				select {
				case fi := <-l.activateInChan:
					l.activateFilter(fi)
				case fi := <-l.backpropInChan:
					l.backpropFilter(fi)
				}
				l.wg.Done()
			}
		}()
	}

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *layer) Activate(inputs *data.Data) *data.Data {
	l.inputs = inputs

	l.wg.Add(len(l.output.Data))
	for i := 0; i < len(l.output.Data); i++ {
		l.activateInChan <- i
	}
	l.wg.Wait()

	return l.output
}

func (l *layer) activateFilter(i int) {
	k := i * l.iVolume
	o := 0.0

	weightsData := l.Weights.Data[k : k+len(l.inputs.Data)]
	inputsData := l.inputs.Data

	for j := 0; j < l.iVolume; j++ {
		o += weightsData[j] * inputsData[j]
	}

	l.output.Data[i] = o + l.Biases.Data[i]
}

func (l *layer) ResetGradients() {
	l.gradWeights.Reset()
	l.gradBiases.Reset()
}

func (l *layer) Backprop(deltas *data.Data) *data.Data {
	l.gradInputs.Reset()
	l.deltas = deltas

	l.wg.Add(len(l.output.Data))
	for i := 0; i < len(l.output.Data); i++ {
		l.backpropInChan <- i
	}
	l.wg.Wait()
	return l.gradInputs
}

func (l *layer) backpropFilter(i int) {
	k := i * l.iVolume
	delta := l.deltas.Data[i]

	weightsData := l.Weights.Data[k : k+len(l.inputs.Data)]
	inputsData := l.inputs.Data

	gradWeightsData := l.gradWeights.Data[k : k+len(l.inputs.Data)]
	gradInputsData := l.gradInputs.Data

	for j := 0; j < l.iVolume; j++ {
		gradInputsData[j] += weightsData[j] * delta
		gradWeightsData[j] += inputsData[j] * delta
	}
	l.gradBiases.Data[i] += delta
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

func (l *layer) GetWeightGradients() *data.Data {
	return l.gradWeights
}

func (l *layer) IsTrainable() bool {
	return l.Trainable
}
