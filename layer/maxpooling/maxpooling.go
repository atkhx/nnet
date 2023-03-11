package maxpooling

import (
	"github.com/atkhx/nnet/data"
)

func New(options ...Option) *MaxPool {
	layer := &MaxPool{}
	applyOptions(layer, defaults...)
	applyOptions(layer, options...)

	layer.oWidth = (layer.iWidth-layer.FWidth+2*layer.FPadding)/layer.FStride + 1
	layer.oHeight = (layer.iHeight-layer.FHeight+2*layer.FPadding)/layer.FStride + 1
	layer.oDepth = layer.iDepth

	return layer
}

type MaxPool struct {
	inputs, output *data.Data

	iWidth, iHeight, iDepth int
	oWidth, oHeight, oDepth int

	FWidth, FHeight   int
	FStride, FPadding int
}

//nolint:gocognit
func (l *MaxPool) Forward(inputs *data.Data) *data.Data {
	l.inputs = inputs

	imagesCount := inputs.Data.D

	l.output = data.NewData(
		l.oWidth*l.oHeight,
		l.oDepth,
		imagesCount,
		make([]float64, l.oWidth*l.oHeight*l.oDepth*imagesCount),
	)

	coords := make([]int, l.oWidth*l.oHeight*l.oDepth*imagesCount)

	oSquare := l.oWidth * l.oHeight
	iSquare := l.iWidth * l.iHeight

	for imageIndex := 0; imageIndex < imagesCount; imageIndex++ {
		imageOffset := imageIndex * iSquare * l.iDepth
		outputOffset := imageIndex * oSquare * l.oDepth

		for oz := 0; oz < l.oDepth; oz++ {
			wW, wH := l.FWidth, l.FHeight
			outXYZ := outputOffset + oz*oSquare
			max := 0.0
			maxCoord := 0

			for oy := 0; oy < l.oHeight; oy++ {
				for ox := 0; ox < l.oWidth; ox++ {
					iy, n := oy*l.FStride-l.FPadding, true

					for fy := 0; fy < wH; fy++ {
						ix := ox*l.FStride - l.FPadding
						for fx := 0; fx < wW; fx++ {
							if ix > -1 && ix < l.iWidth && iy > -1 && iy < l.iHeight {
								inXYZ := imageOffset + oz*iSquare + iy*l.iWidth + ix

								if n || max < inputs.Data.Data[inXYZ] {
									max, maxCoord, n = inputs.Data.Data[inXYZ], inXYZ, false
								}
							}

							ix++
						}
						iy++
					}

					l.output.Data.Data[outXYZ] = max
					coords[outXYZ] = maxCoord

					outXYZ++
				}
			}
		}
	}

	l.output.SetParentsAndBackwardFn([]*data.Data{inputs}, func() {
		for i, coord := range coords {
			inputs.Grad.Data[coord] += l.output.Grad.Data[i]
		}
	})

	return l.output
}

func (l *MaxPool) GetOutput() *data.Data {
	return l.output
}

func (l *MaxPool) GetInputGradients() *data.Volume {
	return l.inputs.Grad
}
