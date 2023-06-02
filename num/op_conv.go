package num

import "sync"

func (aData *Data) Conv(imageWidth, imageHeight, filterSize, padding, stride int, filters, biases *Data) *Data {
	channels := aData.Dims.H
	imagesCount := aData.Dims.D
	filtersCount := filters.Dims.D

	outImageWidth, outImageHeight := CalcConvOutputSize(
		imageWidth, imageHeight,
		filterSize, filterSize,
		padding, stride,
	)

	oSquare := outImageWidth * outImageHeight

	iCube := imageWidth * imageHeight * channels
	fCube := filterSize * filterSize * channels

	wg := sync.WaitGroup{}

	output := New(NewDims(outImageWidth*outImageHeight, filtersCount, imagesCount), aData, filters)
	output.calcData = func() {
		offset := 0
		for imageIndex := 0; imageIndex < imagesCount; imageIndex++ {
			image := aData.Data[imageIndex*iCube : (imageIndex+1)*iCube]

			for filterIndex := 0; filterIndex < filtersCount; filterIndex++ {
				wg.Add(1)
				go func(image Float64s, filterIndex, offset int) {
					featureMap := output.Data[offset : offset+oSquare]
					featureMap.Fill(biases.Data[filterIndex])

					filter := filters.Data[filterIndex*fCube : (filterIndex+1)*fCube]

					ConvTo(
						outImageWidth, outImageHeight, featureMap,
						imageWidth, imageHeight, image,
						filterSize, filterSize, filter,
						channels,
						padding,
					)
					wg.Done()
				}(image, filterIndex, offset)
				offset += oSquare
			}
		}
		wg.Wait()
	}

	output.calcGrad = func() {
		_, _, filtersRotate := Rotate180(filterSize, filterSize, filtersCount*channels, filters.Data)

		inputs := aData
		outputGrad := output.Grad

		offset := 0

		for imageIndex := 0; imageIndex < imagesCount; imageIndex++ {
			iGrads := inputs.Grad[imageIndex*iCube : (imageIndex+1)*iCube]
			inputs := inputs.Data[imageIndex*iCube : (imageIndex+1)*iCube]

			for filterIndex := 0; filterIndex < filtersCount; filterIndex++ {
				wg.Add(1)
				go func(inputs, iGrads Float64s, filterIndex, offset int) {
					deltas := outputGrad[offset : offset+oSquare]
					biases.Grad[filterIndex] += deltas.Sum()

					filtersGrad := filters.Grad[filterIndex*fCube : (filterIndex+1)*fCube]
					filtersRotData := filtersRotate[filterIndex*fCube : (filterIndex+1)*fCube]

					// fWH[D] = iWH[D] x Dy[1]
					// dW     = I      x Dy
					Convolve2dBatchTo(
						filterSize, filterSize, filtersGrad,
						imageWidth, imageHeight, channels, inputs,
						outImageWidth, outImageHeight, 1, deltas,
						padding,
					)

					// iWH[D] = oWoH[1] x fWH[D]
					// dI     = DyPad   x Wrot180
					Convolve2dBatchTo(
						imageWidth, imageHeight, iGrads,
						outImageWidth, outImageHeight, 1, deltas,
						filterSize, filterSize, channels, filtersRotData,
						padding,
					)

					wg.Done()
				}(inputs, iGrads, filterIndex, offset)

				wg.Wait()

				offset += oSquare
			}
		}
	}

	return output
}

func CalcConvOutputSize(
	iw, ih int,
	fw, fh int,
	padding int,
	stride int,
) (int, int) {
	return (iw-fw+2*padding)/stride + 1, (ih-fh+2*padding)/stride + 1
}

func ConvTo(
	ow, oh int, output []float64, // 1d
	iw, ih int, inputs []float64, // depth = channels
	fw, fh int, filter []float64, // depth = channels
	channels int,
	padding int,
) {
	iWiH := iw * ih
	iWHD := iWiH * channels

	wCoord := 0

	for izFrom := 0; izFrom < iWHD; izFrom += iWiH {
		for fy := 0; fy < fh; fy++ {
			oyOffsetLeft := positive(padding - fy)
			oyOffsetRight := positive(fy - padding)

			maxW := min(oh, ih+oyOffsetLeft-oyOffsetRight) * ow

			iyFrom := izFrom + oyOffsetRight*iw
			oyFrom := oyOffsetLeft * ow

			for fx := 0; fx < fw; fx++ {
				weight := filter[wCoord]
				wCoord++

				ox := positive(padding - fx)
				ix := positive(fx - padding)

				OW := min(ow-ox, iw-ix)

				for ixFrom, oxFrom := iyFrom+ix, oyFrom+ox; oxFrom < maxW; ixFrom, oxFrom = ixFrom+iw, oxFrom+ow {
					output := output[oxFrom : oxFrom+OW]
					inputs := inputs[ixFrom : ixFrom+OW]
					for ic, iv := range inputs {
						output[ic] += iv * weight
					}
				}
			}
		}
	}
}

// Convolve2dBatchTo convolve each 2d-input by each 2d-filter and store separate convolution result in output.
func Convolve2dBatchTo(
	ow, oh int, output []float64,
	iw, ih, ic int, inputs []float64,
	fw, fh, fc int, filter []float64,
	padding int,
) {
	outputSquare := ow * oh
	inputsSquare := iw * ih
	filterSquare := fw * fh

	outputOffset := 0
	inputsOffset := 0

	for ii := 0; ii < ic; ii++ {
		filterOffset := 0
		for fi := 0; fi < fc; fi++ {
			ConvTo(
				ow, oh, output[outputOffset:outputOffset+outputSquare], // iOutput
				iw, ih, inputs[inputsOffset:inputsOffset+inputsSquare], // 1d
				fw, fh, filter[filterOffset:filterOffset+filterSquare], // iFilter
				1,
				padding,
			)

			outputOffset += outputSquare
			filterOffset += filterSquare
		}
		inputsOffset += inputsSquare
	}
}

func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}

func positive(f int) int {
	if f > 0 {
		return f
	}
	return 0
}

func Rotate180(iw, ih, id int, a []float64) (ow, oh int, b []float64) {
	b = make([]float64, len(a))
	ow, oh = ih, iw

	for z := 0; z < id; z++ {
		for y := 0; y < ih; y++ {
			for x := 0; x < iw; x++ {
				b[z*iw*ih+(ih-y-1)*iw+(iw-x-1)] = a[z*iw*ih+y*iw+x]
			}
		}
	}

	return
}
