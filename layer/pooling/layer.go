package pooling

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
	oWidth, oHeight, oDepth int

	iSquare int
	oSquare int

	fWidth  int
	fHeight int

	fStride  int
	fPadding int

	inputs *data.Data
	output *data.Data
	coords []int

	deltas     *data.Data
	gradInputs *data.Data

	threads int

	activateInChan chan int
	backpropInChan chan int

	wg sync.WaitGroup
}

func (l *layer) InitDataSizes(w, h, d int) (int, int, int) {
	if l.fStride < 1 {
		l.fStride = 1
	}

	l.iWidth, l.iHeight, l.iDepth = w, h, d

	l.oWidth = (l.iWidth-l.fWidth+2*l.fPadding)/l.fStride + 1
	l.oHeight = (l.iHeight-l.fHeight+2*l.fPadding)/l.fStride + 1
	l.oDepth = l.iDepth

	l.output = &data.Data{}
	l.output.InitCube(l.oWidth, l.oHeight, l.oDepth)

	l.gradInputs = &data.Data{}
	l.gradInputs.InitCube(l.iWidth, l.iHeight, l.iDepth)

	l.iSquare = l.iWidth * l.iHeight
	l.oSquare = l.oWidth * l.oHeight
	l.coords = make([]int, l.oWidth*l.oHeight*l.oDepth)

	if l.threads == 0 {
		l.threads = l.oDepth
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

	l.wg.Add(l.oDepth)
	for i := 0; i < l.oDepth; i++ {
		l.activateInChan <- i
	}
	l.wg.Wait()

	return l.output
}

func (l *layer) activateFilter(oz int) {
	wW, wH := l.fWidth, l.fHeight
	outXYZ := oz * l.oSquare
	max := 0.0
	maxCoord := 0

	for oy := 0; oy < l.oHeight; oy++ {
		for ox := 0; ox < l.oWidth; ox++ {

			iy, n := oy*l.fStride-l.fPadding, true

			for fy := 0; fy < wH; fy++ {
				ix := ox*l.fStride - l.fPadding
				for fx := 0; fx < wW; fx++ {
					if ix > -1 && ix < l.iWidth && iy > -1 && iy < l.iHeight {
						inXYZ := oz*l.iSquare + iy*l.iWidth + ix

						if n || max < l.inputs.Data[inXYZ] {
							max, maxCoord, n = l.inputs.Data[inXYZ], inXYZ, false
						}
					}

					ix++
				}
				iy++
			}

			l.output.Data[outXYZ] = max
			l.coords[outXYZ] = maxCoord

			outXYZ++
		}
	}
}

func (l *layer) Backprop(deltas *data.Data) *data.Data {
	l.gradInputs.Reset()
	l.deltas = deltas

	l.wg.Add(l.oDepth)
	for i := 0; i < l.oDepth; i++ {
		l.backpropInChan <- i
	}
	l.wg.Wait()

	return l.gradInputs
}

func (l *layer) backpropFilter(oz int) {
	offset := oz * l.oSquare
	for i := offset; i < offset+l.oSquare; i++ {
		l.gradInputs.Data[l.coords[i]] += l.deltas.Data[i]
	}
}

func (l *layer) GetOutput() *data.Data {
	return l.output
}

func (l *layer) GetInputGradients() *data.Data {
	return l.gradInputs
}
