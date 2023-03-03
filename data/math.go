package data

import (
	"fmt"
)

func MatrixColFloatsChan(aColsCount, aRowsCount, colIndex, chanIndex int, a []float64) []float64 {
	offset := chanIndex * aColsCount * aRowsCount

	res := make([]float64, aRowsCount)
	for rowIndex := 0; rowIndex < aRowsCount; rowIndex++ {
		res[rowIndex] = a[offset+rowIndex*aColsCount+colIndex]
	}

	return res
}

func MatrixSetColFloatsChan(aColsCount, aRowsCount, colIndex, chanIndex int, a, b []float64) {
	offset := chanIndex*aColsCount*aRowsCount + colIndex

	for rowIndex := 0; rowIndex < aRowsCount; rowIndex++ {
		a[offset+rowIndex*aColsCount] = b[rowIndex]
	}
}

func MatrixColFloats(aColsCount, aRowsCount, colIndex int, a []float64) []float64 {
	res := make([]float64, aRowsCount)
	for rowIndex := 0; rowIndex < aRowsCount; rowIndex++ {
		res[rowIndex] = a[rowIndex*aColsCount+colIndex]
	}

	return res
}

func MatrixSetColFloats(aColsCount, aRowsCount, colIndex int, a, b []float64) {
	for rowIndex := 0; rowIndex < aRowsCount; rowIndex++ {
		a[rowIndex*aColsCount+colIndex] = b[rowIndex]
	}
}

func MatrixRowFloatsChan(aColsCount, aRowsCount, rowIndex, chanIndex int, a []float64) []float64 {
	offset := chanIndex*aColsCount*aRowsCount + rowIndex*aColsCount
	return a[offset : offset+aColsCount]
}

func MatrixRowFloats(aColsCount, rowIndex int, a []float64) []float64 {
	rowOffset := rowIndex * aColsCount
	return a[rowOffset : rowOffset+aColsCount]
}

func MatrixRowFloatsCopy(aColsCount, rowIndex int, a []float64) []float64 {
	return CopyWithData(MatrixRowFloats(aColsCount, rowIndex, a))
}

func MatrixMultiply(
	aColsCount, aRowsCount int, a []float64,
	bColsCount, bRowsCount int, b []float64,
) (
	rColsCount, rRowsCount int, r []float64,
) {
	if aColsCount != bRowsCount {
		panic(fmt.Sprintf("aColsCount != bRowsCount: %d != %d", aColsCount, bRowsCount))
	}

	rColsCount, rRowsCount = bColsCount, aRowsCount

	r = make([]float64, rColsCount*rRowsCount)

	_, _, bT := MatrixTranspose(bColsCount, bRowsCount, b)

	for weightIndex := 0; weightIndex < bColsCount; weightIndex++ {
		//bFloats := MatrixColFloats(bColsCount, bRowsCount, weightIndex, b)

		for inputIndex := 0; inputIndex < aRowsCount; inputIndex++ {
			aFloats := MatrixRowFloats(aColsCount, inputIndex, a)
			bFloats := MatrixRowFloats(bRowsCount, weightIndex, bT)

			r[inputIndex*bColsCount+weightIndex] = Dot(aFloats, bFloats)
		}
	}

	return
}

func MatrixAddRowVector(aColsCount, aRowsCount int, matrix, vec []float64) (out []float64) {
	if aColsCount != len(vec) {
		panic(fmt.Sprintf("invalid vector length: expected %d, actual %d", aColsCount, len(vec)))
	}

	out = make([]float64, 0, aColsCount*aRowsCount)

	for rowIndex := 0; rowIndex < aRowsCount; rowIndex++ {
		out = append(out, Add(MatrixRowFloats(aColsCount, rowIndex, matrix), vec)...)
	}

	return
}

func MatrixTranspose(
	aColsCount, aRowsCount int, a []float64,
) (
	rColsCount, rRowsCount int, r []float64,
) {
	rColsCount, rRowsCount = aRowsCount, aColsCount
	r = make([]float64, rColsCount*rRowsCount)

	for row := 0; row < rRowsCount; row++ {
		for col := 0; col < rColsCount; col++ {
			r[row*rColsCount+col] = a[col*aColsCount+row]
		}
	}
	return
}

func MatrixTransposeChan(
	aColsCount, aRowsCount, aChanCount int, a []float64,
) (
	rColsCount, rRowsCount, rChanCount int, r []float64,
) {
	rColsCount, rRowsCount, rChanCount = aRowsCount, aColsCount, aChanCount
	r = make([]float64, rColsCount*rRowsCount*rChanCount)

	offset := 0
	for c := 0; c < rChanCount; c++ {
		for row := 0; row < rRowsCount; row++ {
			for col := 0; col < rColsCount; col++ {
				r[offset+row*rColsCount+col] = a[offset+col*aColsCount+row]
			}
		}
		offset += rColsCount * rRowsCount
	}
	return
}

func MatrixRotate180(iw, ih int, a []float64) (ow, oh int, b []float64) {
	b = make([]float64, len(a))
	ow, oh = ih, iw

	for y := 0; y < ih; y++ {
		for x := 0; x < iw; x++ {
			b[(ih-y-1)*iw+(iw-x-1)] = a[y*iw+x]
		}
	}
	return
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

func Conv2D(
	iw, ih int, image []float64,
	fw, fh int, filter []float64,
	channels int,
	padding int,
	stride int,
) (
	ow, oh int, output []float64,
) {
	ow = (iw-fw+2*padding)/stride + 1
	oh = (ih-fh+2*padding)/stride + 1

	inputs, iw, ih := AddPadding(image, iw, ih, channels, padding)

	output = make([]float64, ow*oh)

	fHiW := fh * iw
	oHiW := oh * iw

	iSquare := iw * ih
	iCube := iSquare * channels

	wCoord := 0
	for izo := 0; izo < iCube; izo += iSquare {
		for iyo := izo; iyo < izo+fHiW; iyo += iw {
			for ixo := iyo; ixo < iyo+fw; ixo++ {
				weight := filter[wCoord]
				wCoord++

				oCoord := 0
				for iCoord := ixo; iCoord < ixo+oHiW; iCoord += iw {
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

	return
}

func AddPadding(
	src []float64,
	iw, ih int,
	channels int, // todo remove
	padding int,
) ([]float64, int, int) {
	if padding == 0 {
		return src, iw, ih
	}

	var ow, oh, od int
	var pw, ph, pd int

	// extract dimensions
	ow, oh, od = iw, ih, channels

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

	return res, pw, ph
}

func RemovePadding(
	src []float64,
	iw, ih int,
	channels int, // todo remove
	padding int,
) ([]float64, int, int) {
	if padding == 0 {
		return src, iw, ih
	}

	var ow, oh, od int
	var pw, ph, pd int
	ow, oh, od = iw, ih, channels

	pd = od
	pw = ow - 2*padding //nolint:gomnd
	ph = oh - 2*padding //nolint:gomnd

	res := make([]float64, pw*ph*pd)

	for z := 0; z < pd; z++ {
		for y := 0; y < ph; y++ {
			copy(
				res[z*ph*pw+y*pw:z*ph*pw+y*pw+pw],
				src[z*oh*ow+(y+padding)*ow+padding:z*oh*ow+(y+padding)*ow+padding+pw],
			)
		}
	}

	return res, pw, ph
}

func Conv2DOld(
	iw, ih int, inputs []float64,
	fw, fh int, filter []float64,
	padding int,
	stride int,
) (
	ow, oh int, output []float64,
) {
	ow = (iw-fw+2*padding)/stride + 1
	oh = (ih-fh+2*padding)/stride + 1

	output = make([]float64, ow*oh)

	oCord := 0

	for oy, initInputY := 0, -padding; oy < oh; oy, initInputY = oy+1, initInputY+stride {
		for ox, initInputX := 0, -padding; ox < ow; ox, initInputX = ox+1, initInputX+stride {

			for fy, iy := 0, initInputY; fy < fh; fy, iy = fy+1, iy+1 {
				if iy > -1 && iy < ih {
					for fx, ix := 0, initInputX; fx < fw; fx, ix = fx+1, ix+1 {
						if ix > -1 && ix < iw {
							inXYZ := iy*iw + ix
							wtXYZ := fy*fw + fx

							output[oCord] += inputs[inXYZ] * filter[wtXYZ]
						}
					}
				}
			}

			oCord++
		}
	}

	if oCord != len(output) {
		panic("ayaya")
	}

	return
}
