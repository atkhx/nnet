package pooling

import (
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
}

func (l *Layer) InitDataSizes(w, h, d int) (int, int, int) {
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

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *Layer) Activate(inputs *data.Data) *data.Data {
	l.inputs = inputs
	executor.RunParallel(l.oDepth, l.activateFilter)
	return l.output
}

func (l *Layer) activateFilter(oz int) {
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

func (l *Layer) Backprop(deltas *data.Data) *data.Data {
	l.gradInputs.FillZero()
	l.deltas = deltas
	executor.RunParallel(l.oDepth, l.backpropFilter)

	return l.gradInputs
}

func (l *Layer) backpropFilter(oz int) {
	offset := oz * l.oSquare
	for i := offset; i < offset+l.oSquare; i++ {
		l.gradInputs.Data[l.coords[i]] += l.deltas.Data[i]
	}
}

func (l *Layer) GetOutput() *data.Data {
	return l.output
}

func (l *Layer) GetInputGradients() *data.Data {
	return l.gradInputs
}
