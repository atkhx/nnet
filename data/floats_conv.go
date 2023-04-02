package data

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

// Convolve2dBatchTo convolve each 2d-input by each 2d-filter and store separate convolution result in output
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

func Convolve2dBatchTo2(
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

// Convolve2dBatchTo convolve each 2d-input by each 2d-filter and store separate convolution result in output
func Convolve2dBatchTo22(
	ow, oh int, output []float64,
	iw, ih, ic int, inputs []float64,
	fw, fh, fc int, filter []float64,
	padding int,
) {
	iw, ih, ic, inputs = AddPadding(inputs, iw, ih, ic, padding)
	outputSquare := ow * oh
	inputsSquare := iw * ih
	filterSquare := fw * fh

	outputOffset := 0
	inputsOffset := 0

	for ii := 0; ii < ic; ii++ {
		filterOffset := 0
		for fi := 0; fi < fc; fi++ {
			ConvLayerTo(
				ow, oh, output[outputOffset:outputOffset+outputSquare], // iOutput
				iw, ih, inputs[inputsOffset:inputsOffset+inputsSquare], // 1d
				fw, fh, filter[filterOffset:filterOffset+filterSquare], // iFilter
				//padding,
			)

			outputOffset += outputSquare
			filterOffset += filterSquare
		}
		inputsOffset += inputsSquare
	}
}

func Conv2DTo(
	ow, oh int, output []float64, // depth = filtersCount
	iw, ih int, inputs []float64, // depth = 1
	fw, fh int, filter []float64, // depth = filtersCount
	filtersCount int,
	padding int,
) {
	oWoH := ow * oh
	oWHD := oWoH * filtersCount

	wCoord := 0

	for ozFrom := 0; ozFrom < oWHD; ozFrom += oWoH {
		for fy := 0; fy < fh; fy++ {
			oyOffsetLeft := positive(padding - fy)
			oyOffsetRight := positive(fy - padding)

			maxW := min(oh, ih+oyOffsetLeft-oyOffsetRight) * ow

			iyFrom := oyOffsetRight * iw
			oyFrom := oyOffsetLeft * ow

			for fx := 0; fx < fw; fx++ {
				weight := filter[wCoord]
				wCoord++

				ox := positive(padding - fx)
				ix := positive(fx - padding)

				OW := min(ow-ox, iw-ix)

				for ixFrom, oxFrom := iyFrom+ix, ozFrom+oyFrom+ox; oxFrom < ozFrom+maxW; ixFrom, oxFrom = ixFrom+iw, oxFrom+ow {
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
