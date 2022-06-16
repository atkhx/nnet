package conv

import (
	"sync"

	"github.com/atkhx/nnet/data"
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
	iWidth, iHeight, iDepth int
	FWidth, FHeight, FDepth int
	oWidth, oHeight, oDepth int

	FCount   int
	FPadding int
	FStride  int

	Threads int

	Weights *data.Data
	Biases  *data.Data

	inputs *data.Data
	output *data.Data

	gradWeights *data.Data
	gradBiases  *data.Data
	gradInputs  *data.Data

	iSquare int
	oSquare int
	wSquare int
	wCube   int

	deltas *data.Data

	activateInChan chan int
	backpropInChan chan int

	wg sync.WaitGroup
}

func (l *layer) InitDataSizes(iw, ih, id int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = iw, ih, id

	l.oWidth = (iw-l.FWidth+2*l.FPadding)/l.FStride + 1
	l.oHeight = (ih-l.FHeight+2*l.FPadding)/l.FStride + 1

	l.oDepth = l.FCount
	l.FDepth = id

	if l.Weights == nil {
		l.Weights = &data.Data{}
		l.Biases = &data.Data{}
	}

	if len(l.Weights.Data) == 0 {
		l.Weights.InitCubeRandom(l.FWidth, l.FHeight, l.FCount*l.FDepth, -0.7, 0.7)
		l.Biases.InitVector(l.FCount)
		l.Biases.Fill(0.1)
	}

	l.output = &data.Data{}
	l.output.InitCube(l.oWidth, l.oHeight, l.oDepth)

	l.gradBiases = &data.Data{}
	l.gradBiases.InitVector(l.FCount)

	l.gradWeights = &data.Data{}
	l.gradWeights.InitCube(l.FWidth, l.FHeight, l.FCount*l.FDepth)

	l.gradInputs = &data.Data{}
	l.gradInputs.InitCube(l.iWidth, l.iHeight, l.iDepth)

	l.iSquare = l.iWidth * l.iHeight
	l.oSquare = l.oWidth * l.oHeight
	l.wSquare = l.FWidth * l.FHeight
	l.wCube = l.FDepth * l.wSquare

	if l.Threads == 0 {
		l.Threads = l.FCount
	}

	l.activateInChan = make(chan int, l.Threads)
	l.backpropInChan = make(chan int, l.Threads)

	for i := 0; i < l.Threads; i++ {
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
	l.wg.Add(l.FCount)
	for filterIndex := 0; filterIndex < l.FCount; filterIndex++ {
		l.activateInChan <- filterIndex
	}
	l.wg.Wait()
	return l.output
}

func (l *layer) activateFilter(filterIndex int) {
	filterOutputOffset := filterIndex * l.oSquare // can be i = 0..len(output.Data)
	filterWeightsOffset := filterIndex * l.wCube

	for oy, initInputY := 0, -l.FPadding; oy < l.oHeight; oy, initInputY = oy+1, initInputY+l.FStride {
		for ox, initInputX := 0, -l.FPadding; ox < l.oWidth; ox, initInputX = ox+1, initInputX+l.FStride {
			l.output.Data[filterOutputOffset] = l.Biases.Data[filterIndex]

			wCoord := filterWeightsOffset
			for iz := 0; iz < l.iSquare*l.iDepth; iz += l.iSquare {
				iCoord := iz + initInputY*l.iWidth
				for iy := initInputY; iy < initInputY+l.FHeight; iy++ {
					for ix := initInputX; ix < initInputX+l.FWidth; ix++ {
						if iy > -1 && iy < l.iHeight && ix > -1 && ix < l.iWidth {
							l.output.Data[filterOutputOffset] += l.inputs.Data[iCoord+ix] * l.Weights.Data[wCoord]
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
	l.gradWeights.Reset()
	l.gradBiases.Reset()

	l.deltas = deltas
	l.wg.Add(l.FCount)
	for filterIndex := 0; filterIndex < l.FCount; filterIndex++ {
		l.backpropInChan <- filterIndex
	}
	l.wg.Wait()
	return l.gradInputs
}

func (l *layer) backpropFilter(filterIndex int) {
	filterOutputOffset := filterIndex * l.oSquare // can be i = 0..len(output.Data)
	filterWeightsOffset := filterIndex * l.wCube

	for oy, initInputY := 0, -l.FPadding; oy < l.oHeight; oy, initInputY = oy+1, initInputY+l.FStride {
		for ox, initInputX := 0, -l.FPadding; ox < l.oWidth; ox, initInputX = ox+1, initInputX+l.FStride {

			delta := l.deltas.Data[filterOutputOffset]

			wCoord := filterWeightsOffset
			for iz := 0; iz < l.iSquare*l.iDepth; iz += l.iSquare {
				iCoord := iz + initInputY*l.iWidth
				for iy := initInputY; iy < initInputY+l.FHeight; iy++ {
					for ix := initInputX; ix < initInputX+l.FWidth; ix++ {
						if iy > -1 && iy < l.iHeight && ix > -1 && ix < l.iWidth {
							l.gradInputs.Data[iCoord+ix] += l.Weights.Data[wCoord] * delta
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
	return l.Weights, l.gradWeights
}

func (l *layer) GetBiasesWithGradient() (*data.Data, *data.Data) {
	return l.Biases, l.gradBiases
}

func (l *layer) GetInputGradients() *data.Data {
	return l.gradInputs
}
