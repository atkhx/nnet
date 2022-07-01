package conv

import (
	"fmt"
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
	wSquare int
	wCube   int

	trainable bool

	deltas *data.Data

	threads        int
	activateInChan chan int
	backpropInChan chan int

	wg sync.WaitGroup
}

func (l *layer) InitDataSizes(iw, ih, id int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = iw, ih, id

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
	l.wSquare = l.FWidth * l.FHeight
	l.wCube = l.FDepth * l.wSquare
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
					l.activateFilter(fi)
				case fi := <-l.backpropInChan:
					l.backpropFilter(fi)
				}
				l.wg.Done()
			}
		}()
	}

	fmt.Println("is trainable conv layer", l.IsTrainable())

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *layer) Activate(inputs *data.Data) *data.Data {
	l.inputs = inputs
	l.wg.Add(l.fCount)
	for filterIndex := 0; filterIndex < l.fCount; filterIndex++ {
		l.activateInChan <- filterIndex
	}
	l.wg.Wait()
	return l.output
}

func (l *layer) activateFilter(filterIndex int) {
	filterOutputOffset := filterIndex * l.oSquare
	filterWeightsOffset := filterIndex * l.wCube

	for oy, initInputY := 0, -l.fPadding; oy < l.oHeight; oy, initInputY = oy+1, initInputY+l.fStride {
		for ox, initInputX := 0, -l.fPadding; ox < l.oWidth; ox, initInputX = ox+1, initInputX+l.fStride {
			l.output.Data[filterOutputOffset] = l.biases.Data[filterIndex]

			wCoord := filterWeightsOffset
			for iz := 0; iz < l.iCube; iz += l.iSquare {
				iCoord := iz + initInputY*l.iWidth
				for iy := initInputY; iy < initInputY+l.FHeight; iy++ {
					for ix := initInputX; ix < initInputX+l.FWidth; ix++ {
						if iy > -1 && iy < l.iHeight && ix > -1 && ix < l.iWidth {
							l.output.Data[filterOutputOffset] += l.inputs.Data[iCoord+ix] * l.weights.Data[wCoord]
						}
						wCoord++
					}
					iCoord += l.iWidth
				}
			}
			filterOutputOffset++
		}
	}
}

func (l *layer) Backprop(deltas *data.Data) *data.Data {
	l.gradInputs.Reset()

	l.deltas = deltas
	l.wg.Add(l.fCount)
	for filterIndex := 0; filterIndex < l.fCount; filterIndex++ {
		l.backpropInChan <- filterIndex
	}
	l.wg.Wait()
	return l.gradInputs
}

func (l *layer) backpropFilter(filterIndex int) {
	filterOutputOffset := filterIndex * l.oSquare
	filterWeightsOffset := filterIndex * l.wCube

	for oy, initInputY := 0, -l.fPadding; oy < l.oHeight; oy, initInputY = oy+1, initInputY+l.fStride {
		for ox, initInputX := 0, -l.fPadding; ox < l.oWidth; ox, initInputX = ox+1, initInputX+l.fStride {
			delta := l.deltas.Data[filterOutputOffset]
			wCoord := filterWeightsOffset
			for iz := 0; iz < l.iSquare*l.iDepth; iz += l.iSquare {
				iCoord := iz + initInputY*l.iWidth
				for iy := initInputY; iy < initInputY+l.FHeight; iy++ {
					for ix := initInputX; ix < initInputX+l.FWidth; ix++ {
						if iy > -1 && iy < l.iHeight && ix > -1 && ix < l.iWidth {
							l.gradInputs.Data[iCoord+ix] += l.weights.Data[wCoord] * delta
							l.gradWeights.Data[wCoord] += l.inputs.Data[iCoord+ix] * delta
						}
						wCoord++
					}
					iCoord += l.iWidth
				}
			}

			l.gradBiases.Data[filterIndex] += delta
			filterOutputOffset++
		}
	}
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
