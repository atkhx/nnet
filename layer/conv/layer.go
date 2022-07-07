package conv

import (
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
	iWidth, iHeight, iDepth int
	FWidth, FHeight, FDepth int
	oWidth, oHeight, oDepth int

	fCount   int
	fPadding int
	fStride  int

	InitWeights InitWeightsParams

	weightsRot *data.Data

	weights *data.Data
	biases  *data.Data
	inputs  *data.Data
	output  *data.Data

	gradWeights *data.Data
	gradBiases  *data.Data
	gradInputs  *data.Data

	iSquare int
	iCube   int

	oSquare int
	fSquare int
	fCube   int

	trainable bool

	deltas *data.Data

	threads        int
	activateInChan chan int
	backpropInChan chan int

	wg sync.WaitGroup
}

func (l *layer) InitDataSizes(iw, ih, id int) (int, int, int) {
	//l.iWidth, l.iHeight, l.iDepth = iw, ih, id
	l.iWidth, l.iHeight, l.iDepth = iw+2*l.fPadding, ih+2*l.fPadding, id

	l.oWidth = (iw-l.FWidth+2*l.fPadding)/l.fStride + 1
	l.oHeight = (ih-l.FHeight+2*l.fPadding)/l.fStride + 1

	l.oDepth = l.fCount
	l.FDepth = id

	if l.weights == nil {
		l.weights = &data.Data{}
		l.biases = &data.Data{}
	}

	if len(l.weights.Data) == 0 {
		l.weights.InitCubeRandom(
			l.FWidth,
			l.FHeight,
			l.fCount*l.FDepth,
			l.InitWeights.WeightMinThreshold,
			l.InitWeights.WeightMaxThreshold,
		)
		l.biases.InitVector(l.fCount)
		l.biases.Fill(l.InitWeights.BiasInitialValue)
	}

	l.output = &data.Data{}
	l.output.InitCube(l.oWidth, l.oHeight, l.oDepth)

	l.gradBiases = &data.Data{}
	l.gradBiases.InitVector(l.fCount)

	l.gradWeights = &data.Data{}
	l.gradWeights.InitCube(l.FWidth, l.FHeight, l.fCount*l.FDepth)

	l.gradInputs = &data.Data{}
	l.gradInputs.InitCube(l.iWidth, l.iHeight, l.iDepth)

	l.iSquare = l.iWidth * l.iHeight
	l.oSquare = l.oWidth * l.oHeight
	l.fSquare = l.FWidth * l.FHeight

	l.fCube = l.FDepth * l.fSquare
	l.iCube = l.iDepth * l.iSquare

	if l.threads == 0 {
		l.threads = l.fCount
	}

	l.activateInChan = make(chan int, l.threads)
	l.backpropInChan = make(chan int, l.threads)

	for i := 0; i < l.threads; i++ {
		go func() {
			for {
				select {
				case fi := <-l.activateInChan:
					outputOffset := fi * l.oSquare
					for i := outputOffset; i < outputOffset+l.oSquare; i++ {
						l.output.Data[i] = l.biases.Data[fi]
					}

					data.ConvPadded(
						l.iWidth, l.iHeight, l.iDepth,
						l.oWidth, l.oHeight,
						l.FWidth, l.FHeight,
						l.inputs.Data,
						l.output.Data[outputOffset:outputOffset+l.oSquare],
						l.weights.Data[fi*l.fCube:fi*l.fCube+l.fCube],
					)
				case fi := <-l.backpropInChan:
					data.BackpropConvPadded(
						l.iWidth, l.iHeight, l.iDepth,
						l.oWidth, l.oHeight,
						l.FWidth, l.FHeight,
						l.inputs.Data,
						l.deltas.Data[fi*l.oSquare:fi*l.oSquare+l.oSquare],
						l.weightsRot.Data[fi*l.fCube:fi*l.fCube+l.fCube],
						l.gradInputs.Data,
						l.gradWeights.Data[fi*l.fCube:fi*l.fCube+l.fCube],
					)
				}
				l.wg.Done()
			}
		}()
	}

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *layer) Activate(inputs *data.Data) *data.Data {
	l.inputs = inputs.AddPadding(l.fPadding)

	l.wg.Add(l.fCount)
	for filterIndex := 0; filterIndex < l.fCount; filterIndex++ {
		l.activateInChan <- filterIndex
	}
	l.wg.Wait()

	return l.output
}

func (l *layer) Backprop(deltas *data.Data) *data.Data {
	l.gradInputs.Reset()

	l.weightsRot = l.weights.Copy()
	l.weightsRot.Rotate180()

	l.deltas = deltas

	l.wg.Add(l.fCount)
	for filterIndex := 0; filterIndex < l.fCount; filterIndex++ {
		l.backpropInChan <- filterIndex
	}

	for filterIndex := 0; filterIndex < l.fCount; filterIndex++ {
		for i := filterIndex * l.oSquare; i < (1+filterIndex)*l.oSquare; i++ {
			l.gradBiases.Data[filterIndex] += deltas.Data[i]
		}
	}
	l.wg.Wait()
	return l.gradInputs.RemovePadding(l.fPadding)
}

func (l *layer) ResetGradients() {
	l.gradWeights.Reset()
	l.gradBiases.Reset()
}

func (l *layer) GetWeights() *data.Data {
	return l.weights
}

func (l *layer) GetOutput() *data.Data {
	return l.output
}

func (l *layer) GetInputs() *data.Data {
	return l.inputs
}

func (l *layer) GetWeightsWithGradient() (*data.Data, *data.Data) {
	return l.weights, l.gradWeights
}

func (l *layer) GetBiasesWithGradient() (*data.Data, *data.Data) {
	return l.biases, l.gradBiases
}

func (l *layer) GetInputGradients() *data.Data {
	return l.gradInputs
}

func (l *layer) GetWeightGradients() *data.Data {
	return l.gradWeights
}

func (l *layer) IsTrainable() bool {
	return l.trainable
}
