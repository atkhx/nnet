package maxpooling

import (
	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/executor"
)

func New(options ...Option) *MaxPool {
	layer := &MaxPool{}
	applyOptions(layer, defaults...)
	applyOptions(layer, options...)
	return layer
}

type MaxPool struct {
	iWidth, iHeight, iDepth int
	oWidth, oHeight, oDepth int

	iSquare int
	oSquare int

	FWidth  int
	FHeight int

	FStride  int
	FPadding int

	inputs *data.Data
	output *data.Data
	coords []int

	deltas     *data.Data
	gradInputs *data.Data
}

func (l *MaxPool) InitDataSizes(w, h, d int) (int, int, int) {
	if l.FStride < 1 {
		l.FStride = 1
	}

	l.iWidth, l.iHeight, l.iDepth = w, h, d

	l.oWidth = (l.iWidth-l.FWidth+2*l.FPadding)/l.FStride + 1
	l.oHeight = (l.iHeight-l.FHeight+2*l.FPadding)/l.FStride + 1
	l.oDepth = l.iDepth

	l.output = &data.Data{}
	l.output.Init3D(l.oWidth, l.oHeight, l.oDepth)

	l.gradInputs = &data.Data{}
	l.gradInputs.Init3D(l.iWidth, l.iHeight, l.iDepth)

	l.iSquare = l.iWidth * l.iHeight
	l.oSquare = l.oWidth * l.oHeight
	l.coords = make([]int, l.oWidth*l.oHeight*l.oDepth)

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *MaxPool) Forward(inputs *data.Data) *data.Data {
	l.inputs = inputs
	executor.RunParallel(l.oDepth, func(oz int) {
		wW, wH := l.FWidth, l.FHeight
		outXYZ := oz * l.oSquare
		max := 0.0
		maxCoord := 0

		for oy := 0; oy < l.oHeight; oy++ {
			for ox := 0; ox < l.oWidth; ox++ {

				iy, n := oy*l.FStride-l.FPadding, true

				for fy := 0; fy < wH; fy++ {
					ix := ox*l.FStride - l.FPadding
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
	})
	return l.output
}

func (l *MaxPool) Backward(deltas *data.Data) *data.Data {
	l.gradInputs.FillZero()
	l.deltas = deltas
	executor.RunParallel(l.oDepth, func(oz int) {
		offset := oz * l.oSquare
		for i := offset; i < offset+l.oSquare; i++ {
			l.gradInputs.Data[l.coords[i]] += l.deltas.Data[i]
		}
	})

	return l.gradInputs
}

func (l *MaxPool) GetOutput() *data.Data {
	return l.output
}

func (l *MaxPool) GetInputGradients() *data.Data {
	return l.gradInputs
}
