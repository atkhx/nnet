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
	// begin storable layer config
	FCount   int
	FPadding int
	FStride  int

	Weights *data.Data
	Biases  *data.Data

	GradWeights *data.Data
	GradBiases  *data.Data

	Trainable bool
	// end storable layer config

	iWidth, iHeight, iDepth int
	FWidth, FHeight, FDepth int
	oWidth, oHeight, oDepth int

	initWeights InitWeightsParams

	inputs *data.Data
	output *data.Data
	deltas *data.Data

	gradInputs     *data.Data
	weightsRotated *data.Data

	iSquare int
	oSquare int
	fSquare int

	iCube int
	fCube int

	oiHW int
	fiHW int

	threads        int
	activateInChan chan int
	backpropInChan chan int

	wg sync.WaitGroup
}

func (l *layer) InitDataSizes(iw, ih, id int) (int, int, int) {
	//l.iWidth, l.iHeight, l.iDepth = iw, ih, id
	l.iWidth, l.iHeight, l.iDepth = iw+2*l.FPadding, ih+2*l.FPadding, id

	l.oWidth = (iw-l.FWidth+2*l.FPadding)/l.FStride + 1
	l.oHeight = (ih-l.FHeight+2*l.FPadding)/l.FStride + 1

	l.oDepth = l.FCount
	l.FDepth = id

	l.iSquare = l.iWidth * l.iHeight
	l.oSquare = l.oWidth * l.oHeight
	l.fSquare = l.FWidth * l.FHeight

	l.fCube = l.FDepth * l.fSquare
	l.iCube = l.iDepth * l.iSquare

	l.oiHW = l.oHeight * l.iWidth
	l.fiHW = l.FHeight * l.iWidth

	if l.Weights == nil {
		l.Weights = &data.Data{}
		l.Biases = &data.Data{}
	}

	if len(l.Weights.Data) == 0 {
		l.Weights.InitCubeRandom(
			l.FWidth,
			l.FHeight,
			l.FCount*l.FDepth,
			l.initWeights.WeightMinThreshold,
			l.initWeights.WeightMaxThreshold,
		)
		l.Biases.InitVector(l.FCount)
		l.Biases.Fill(l.initWeights.BiasInitialValue)
	}

	l.output = &data.Data{}
	l.output.InitCube(l.oWidth, l.oHeight, l.oDepth)

	l.GradBiases = &data.Data{}
	l.GradBiases.InitVector(l.FCount)

	l.GradWeights = &data.Data{}
	l.GradWeights.InitCube(l.FWidth, l.FHeight, l.FCount*l.FDepth)

	l.gradInputs = &data.Data{}
	l.gradInputs.InitCube(l.iWidth, l.iHeight, l.iDepth)

	l.listenChannels()

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *layer) Activate(inputs *data.Data) *data.Data {
	l.inputs = inputs.AddPadding(l.FPadding)

	l.wg.Add(l.FCount)
	for filterIndex := 0; filterIndex < l.FCount; filterIndex++ {
		l.activateInChan <- filterIndex
	}
	l.wg.Wait()

	return l.output
}

func (l *layer) Backprop(deltas *data.Data) *data.Data {
	l.gradInputs.Reset()

	l.weightsRotated = l.Weights.Copy()
	l.weightsRotated.Rotate180()

	l.deltas = deltas

	l.wg.Add(l.FCount)
	for filterIndex := 0; filterIndex < l.FCount; filterIndex++ {
		l.backpropInChan <- filterIndex
	}
	l.wg.Wait()
	return l.gradInputs.RemovePadding(l.FPadding)
}

func (l *layer) listenChannels() {
	if l.threads == 0 {
		l.threads = l.FCount
	}

	l.activateInChan = make(chan int, l.threads)
	l.backpropInChan = make(chan int, l.threads)

	for i := 0; i < l.threads; i++ {
		go func() {
			for {
				select {
				case fi := <-l.activateInChan:
					outputOffset := fi * l.oSquare
					filterOffset := fi * l.fCube
					l.activateFilter(
						fi,
						l.inputs.Data,
						l.output.Data[outputOffset:outputOffset+l.oSquare],
						l.Weights.Data[filterOffset:filterOffset+l.fCube],
					)
				case fi := <-l.backpropInChan:
					outputOffset := fi * l.oSquare
					filterOffset := fi * l.fCube
					l.backpropFilter(
						fi,
						l.inputs.Data,
						l.deltas.Data[outputOffset:outputOffset+l.oSquare],
						l.weightsRotated.Data[filterOffset:filterOffset+l.fCube],
						l.gradInputs.Data,
						l.GradWeights.Data[filterOffset:filterOffset+l.fCube],
					)
				}
				l.wg.Done()
			}
		}()
	}
}

func (l *layer) activateFilter(filterIndex int, inputs, output, filter []float64) {
	for i := 0; i < l.oSquare; i++ {
		output[i] = l.Biases.Data[filterIndex]
	}

	wCoord := 0
	for izo := 0; izo < l.iCube; izo += l.iSquare {
		for iyo := izo; iyo < izo+l.fiHW; iyo += l.iWidth {
			for ixo := iyo; ixo < iyo+l.FWidth; ixo++ {
				weight := filter[wCoord]
				oCoord := 0
				for iy := ixo; iy < ixo+l.oiHW; iy += l.iWidth {
					for iCoord := iy; iCoord < iy+l.oWidth; iCoord++ {
						output[oCoord] += inputs[iCoord] * weight
						oCoord++
					}
				}

				wCoord++
			}
		}
	}
}

func (l *layer) backpropFilter(
	filterIndex int,
	inputs,
	deltas,
	filter,
	gradInputs,
	gradFilter []float64,
) {
	for i := 0; i < l.oSquare; i++ {
		l.GradBiases.Data[filterIndex] += deltas[i]
	}

	wCoord := 0
	for izo := 0; izo < l.iCube; izo += l.iSquare {
		for iyo := izo; iyo < izo+l.fiHW; iyo += l.iWidth {
			for ixo := iyo; ixo < iyo+l.FWidth; ixo++ {
				weight := filter[wCoord]
				oCoord := 0
				for iy := ixo; iy < ixo+l.oiHW; iy += l.iWidth {
					for iCoord := iy; iCoord < iy+l.oWidth; iCoord++ {
						gradInputs[iCoord] += deltas[oCoord] * weight
						gradFilter[wCoord] += inputs[iCoord] * deltas[oCoord]

						oCoord++
					}
				}

				wCoord++
			}
		}
	}
}

func (l *layer) ResetGradients() {
	l.GradWeights.Reset()
	l.GradBiases.Reset()
}

func (l *layer) GetWeights() *data.Data {
	return l.Weights
}

func (l *layer) GetOutput() *data.Data {
	return l.output
}

func (l *layer) GetInputs() *data.Data {
	return l.inputs
}

func (l *layer) GetWeightsWithGradient() (*data.Data, *data.Data) {
	return l.Weights, l.GradWeights
}

func (l *layer) GetBiasesWithGradient() (*data.Data, *data.Data) {
	return l.Biases, l.GradBiases
}

func (l *layer) GetInputGradients() *data.Data {
	return l.gradInputs
}

func (l *layer) GetWeightGradients() *data.Data {
	return l.GradWeights
}

func (l *layer) IsTrainable() bool {
	return l.Trainable
}
