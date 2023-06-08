package num

func (aData *Data) MaxPooling(iWidth, iHeight, fSize, padding, stride int) *Data {
	imagesCount := aData.Dims.D
	channels := aData.Dims.H

	oWidth := (iWidth-fSize+2*padding)/stride + 1
	oHeight := (iHeight-fSize+2*padding)/stride + 1

	iSquare := iWidth * iHeight
	oSquare := oWidth * oHeight

	output := New(NewDims(oWidth*oHeight, channels, imagesCount), aData)
	coords := make([]int, oWidth*oHeight*channels*imagesCount)

	output.Name = "maxPooling"
	output.calcData = func() {
		for imageIndex := 0; imageIndex < imagesCount; imageIndex++ {
			imageOffset := imageIndex * iSquare * channels
			outputOffset := imageIndex * oSquare * channels

			for oz := 0; oz < channels; oz++ {
				wW, wH := fSize, fSize
				outXYZ := outputOffset + oz*oSquare
				max := 0.0
				maxCoord := 0

				for oy := 0; oy < oHeight; oy++ {
					for ox := 0; ox < oWidth; ox++ {
						iy, n := oy*stride-padding, true

						for fy := 0; fy < wH; fy++ {
							ix := ox*stride - padding
							for fx := 0; fx < wW; fx++ {
								if ix > -1 && ix < iWidth && iy > -1 && iy < iHeight {
									inXYZ := imageOffset + oz*iSquare + iy*iWidth + ix

									if n || max < aData.Data[inXYZ] {
										max, maxCoord, n = aData.Data[inXYZ], inXYZ, false
									}
								}

								ix++
							}
							iy++
						}

						output.Data[outXYZ] = max
						coords[outXYZ] = maxCoord

						outXYZ++
					}
				}
			}
		}
	}

	output.calcGrad = func() {
		for i, coord := range coords {
			aData.Grad[coord] += output.Grad[i]
		}
	}

	return output
}
