package pooling

import (
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
	oWidth, oHeight, oDepth int

	FWidth  int
	FHeight int

	FStride  int
	FPadding int

	inputs *data.Data
	output *data.Data
	coords []int

	gradInputs *data.Data
}

func (l *layer) InitDataSizes(w, h, d int) (int, int, int) {
	if l.FStride < 1 {
		l.FStride = 1
	}

	l.iWidth, l.iHeight, l.iDepth = w, h, d

	l.oWidth = (l.iWidth-l.FWidth+2*l.FPadding)/l.FStride + 1
	l.oHeight = (l.iHeight-l.FHeight+2*l.FPadding)/l.FStride + 1
	l.oDepth = l.iDepth

	l.output = &data.Data{}
	l.output.InitCube(l.oWidth, l.oHeight, l.oDepth)

	l.gradInputs = &data.Data{}
	l.gradInputs.InitCube(l.iWidth, l.iHeight, l.iDepth)

	l.coords = make([]int, l.oWidth*l.oHeight*l.oDepth)

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *layer) Activate(inputs *data.Data) *data.Data {
	l.inputs = inputs

	wW, wH := l.FWidth, l.FHeight

	outXYZ := 0
	iSquare := l.iWidth * l.iHeight

	max := 0.0
	maxCoord := 0
	for oz := 0; oz < l.oDepth; oz++ {
		for oy := 0; oy < l.oHeight; oy++ {
			for ox := 0; ox < l.oWidth; ox++ {

				iy, n := oy*l.FStride-l.FPadding, true

				for fy := 0; fy < wH; fy++ {
					ix := ox*l.FStride - l.FPadding
					for fx := 0; fx < wW; fx++ {
						if ix > -1 && ix < l.iWidth && iy > -1 && iy < l.iHeight {
							inXYZ := oz*iSquare + iy*l.iWidth + ix

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
	return l.output
}

func (l *layer) Backprop(deltas *data.Data) *data.Data {
	l.gradInputs.Reset()

	for i := 0; i < len(deltas.Data); i++ {
		l.gradInputs.Data[l.coords[i]] += deltas.Data[i]
	}
	return l.gradInputs
}

func (l *layer) GetOutput() *data.Data {
	return l.output
}

func (l *layer) GetInputGradients() *data.Data {
	return l.gradInputs
}
