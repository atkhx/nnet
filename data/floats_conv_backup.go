package data

import "fmt"

func Conv3DPaddedTo(
	ow, oh int, output []float64,
	iw, ih int, inputs []float64,
	fw, fh int, filter []float64,
	channels int,
) {
	// todo implement stride param
	fHiW := fh * iw
	oHiW := oh * iw

	iSquare := iw * ih
	iCube := iSquare * channels

	wCoord := 0
	for izFrom := 0; izFrom < iCube; izFrom += iSquare {
		for iyFrom := izFrom; iyFrom < izFrom+fHiW; iyFrom += iw {
			for ixFrom := iyFrom; ixFrom < iyFrom+fw; ixFrom++ {
				weight := filter[wCoord]
				wCoord++

				for iCoord, oCoord := ixFrom, 0; iCoord < ixFrom+oHiW; iCoord += iw {
					output := output[oCoord : oCoord+ow]
					inputs := inputs[iCoord : iCoord+ow]
					for ic, iv := range inputs {
						output[ic] += iv * weight
					}
					oCoord += ow
				}
			}
		}
	}
}

func ConvLayerTo(
	ow, oh int, output []float64,
	iw, ih int, inputs []float64,
	fw, fh int, filter []float64,
) {
	oc := 0
	for oy := 0; oy < oh; oy++ {
		for ox := 0; ox < ow; ox++ {

			ov := 0.0
			fc := 0
			for fy := 0; fy < fh; fy++ {
				for fx := 0; fx < fw; fx++ {
					ic := (oy+fy)*iw + (ox + fx)
					ov += inputs[ic] * filter[fc]
					fc++
				}
			}

			output[oc] += ov
			oc++
		}
	}
}

func ConvolveTo2WorksForPadding1(
	ow, oh int, output []float64,
	iw, ih int, inputs []float64,
	fw, fh int, filter []float64,
	channels int,
	padding int,
) {
	// todo implement stride param
	//fHiW := fh * iw
	//oHiW := oh * iw

	iSquare := iw * ih
	//iCube := iSquare * channels

	wCoord := 0

	for fz := 0; fz < channels; fz++ {
		for fy := 0; fy < fh; fy++ {
			for fx := 0; fx < fw; fx++ {
				weight := filter[wCoord]
				if false {
					fmt.Println("weight", weight)
				}
				wCoord++

				oyOffset := positive(padding - fy)
				oxOffset := positive(padding - fx)

				for oy := oyOffset; oy < oh-positive(fy-padding); oy++ {
					ox := oxOffset
					OW := ow - ox - positive(fx-padding)

					iy := oy - positive(padding-fy) + positive(fy-padding)
					ix := positive(fx - padding)

					fmt.Println("oy", oy, "ox", ox, "OW", OW, "|", "iy", iy, "ix", ix)

					output := output[oy*ow+ox : oy*ow+ox+OW]
					inputs := inputs[fz*iSquare+iy*iw+ix : fz*iSquare+iy*iw+ix+OW]

					//fmt.Println("output", output)
					//fmt.Println("inputs", inputs)

					if len(inputs) != len(output) {
						panic("invalid length")
					}

					for ic, iv := range inputs {
						output[ic] += iv * weight
					}
				}
				fmt.Println()
			}
			fmt.Println("----")
		}
	}
}

func AddPadding(src []float64, iw, ih, id, padding int) (int, int, int, []float64) {
	if padding == 0 {
		return iw, ih, id, src
	}

	var ow, oh, od int
	var pw, ph, pd int

	// extract dimensions
	ow, oh, od = iw, ih, id

	pd = od
	pw = ow + 2*padding //nolint:gomnd
	ph = oh + 2*padding //nolint:gomnd

	res := make([]float64, pw*ph*pd)

	phpw := ph * pw
	ohow := oh * ow

	for z := 0; z < pd; z++ {
		for y := padding; y < ph-padding; y++ {
			copy(
				res[z*phpw+y*pw+padding:z*phpw+y*pw+padding+ow],
				src[z*ohow+(y-padding)*ow:z*ohow+(y-padding)*ow+ow],
			)
		}
	}

	return pw, ph, pd, res
}

func ConvolveTo2WorksAlways(
	ow, oh int, output []float64,
	iw, ih int, inputs []float64,
	fw, fh int, filter []float64,
	channels int,
	padding int,
) {
	// todo implement stride param
	//fHiW := fh * iw
	//oHiW := oh * iw

	iSquare := iw * ih
	//iCube := iSquare * channels

	wCoord := 0

	for fz := 0; fz < channels; fz++ {
		for fy := 0; fy < fh; fy++ {
			for fx := 0; fx < fw; fx++ {
				weight := filter[wCoord]
				if false {
					fmt.Println("weight", weight)
				}
				wCoord++

				oyOffset := positive(padding - fy)
				oxOffset := positive(padding - fx)

				//for oy := oyOffset; oy < oh-positive(fy-padding); oy++ {
				//for oy := oyOffset; oy < oyOffset+ih; oy++ {
				//for oy := oyOffset; oy < oh; oy++ {
				for oy := oyOffset; oy < min(oh, ih+oyOffset-positive(fy-padding)); oy++ {
					ox := oxOffset
					ix := positive(fx - padding)
					///OW := ow - ox - positive(fx-padding)
					OW := iw // - padding - fx
					OW = min(ow-ox, iw-ix)

					iy := oy - positive(padding-fy) + positive(fy-padding)

					fmt.Println("oy", oy, "ox", ox, "OW", OW, "|", "iy", iy, "ix", ix)

					output := output[oy*ow+ox : oy*ow+ox+OW]
					inputs := inputs[fz*iSquare+iy*iw+ix : fz*iSquare+iy*iw+ix+OW]

					//fmt.Println("output", output)
					//fmt.Println("inputs", inputs)

					if len(inputs) != len(output) {
						panic("invalid length")
					}

					for ic, iv := range inputs {
						output[ic] += iv * weight
					}
				}
				fmt.Println()
			}
			fmt.Println("----")
		}
	}
}
