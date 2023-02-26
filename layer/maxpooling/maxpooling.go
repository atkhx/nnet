package maxpooling

import (
	"github.com/atkhx/nnet/data"
)

func New(options ...Option) *MaxPool {
	layer := &MaxPool{}
	applyOptions(layer, defaults...)
	applyOptions(layer, options...)
	return layer
}

type MaxPool struct {
	inputs, output *data.Matrix

	coords []int

	iWidth, iHeight, iDepth int
	oWidth, oHeight, oDepth int

	FWidth, FHeight   int
	FStride, FPadding int
}

//nolint:gocognit
func (l *MaxPool) Forward(inputs *data.Matrix) *data.Matrix {
	l.inputs = inputs
	l.output = data.NewMatrix(l.oWidth, l.oHeight*l.oDepth, make([]float64, l.oWidth*l.oHeight*l.oDepth))

	oSquare := l.oWidth * l.oHeight
	iSquare := l.iWidth * l.iHeight

	for oz := 0; oz < l.oDepth; oz++ {
		wW, wH := l.FWidth, l.FHeight
		outXYZ := oz * oSquare
		max := 0.0
		maxCoord := 0

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

	l.output.From = data.NewSource(func() {
		l.inputs.InitGrad()
		for oz := 0; oz < l.oDepth; oz++ {
			offset := oz * oSquare
			for i := offset; i < offset+oSquare; i++ {
				l.inputs.Grad[l.coords[i]] += l.output.Grad[i]
			}
		}
	}, l.inputs)

	return l.output
}

func (l *MaxPool) GetOutput() *data.Matrix {
	return l.output
}

func (l *MaxPool) GetInputGradients() *data.Matrix {
	return l.inputs.GradsMatrix()
}
