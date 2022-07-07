package data

func ConvPadded(
	iWidth, iHeight, iDepth,
	oWidth, oHeight,
	fWidth, fHeight int,
	inputs,
	output,
	filter []float64,
) {
	wCoord := 0
	oiHW := oHeight * iWidth
	fiHW := fHeight * iWidth
	for izo := 0; izo < iWidth*iHeight*iDepth; izo += iWidth * iHeight {
		for iyo := izo; iyo < izo+fiHW; iyo += iWidth {
			for ixo := iyo; ixo < iyo+fWidth; ixo++ {
				weight := filter[wCoord]
				oCoord := 0
				for iy := ixo; iy < ixo+oiHW; iy += iWidth {
					for ix := iy; ix < iy+oWidth; ix++ {
						output[oCoord] += inputs[ix] * weight
						oCoord++
					}
				}

				wCoord++
			}
		}
	}
}

func BackpropConvPadded(
	iWidth, iHeight, iDepth,
	oWidth, oHeight,
	fWidth, fHeight int,
	inputs,
	deltas,
	filter,
	gradInputs,
	gradFilter []float64,
) {
	wCoord := 0
	oiHW := oHeight * iWidth
	fiHW := fHeight * iWidth
	for izo := 0; izo < iWidth*iHeight*iDepth; izo += iWidth * iHeight {
		for iyo := izo; iyo < izo+fiHW; iyo += iWidth {
			for ixo := iyo; ixo < iyo+fWidth; ixo++ {
				weight := filter[wCoord]
				oCoord := 0
				for iy := ixo; iy < ixo+oiHW; iy += iWidth {
					for ix := iy; ix < iy+oWidth; ix++ {
						gradInputs[ix] += deltas[oCoord] * weight
						//gi[ix] += deltas[oCoord] * weight
						gradFilter[wCoord] += deltas[oCoord] * inputs[ix]

						oCoord++
					}
				}

				wCoord++
			}
		}
	}
}
