package data

import (
	"math"
	"math/rand"
)

func Copy(src []float64) (dst []float64) {
	dst = make([]float64, len(src))
	copy(dst, src)
	return
}

func Fill(dst []float64, v float64) {
	for k := range dst {
		dst[k] = v
	}
}

func FillRandom(dst []float64) {
	for i := range dst {
		dst[i] = rand.Float64()
	}
}

func MakeRandom(size int) (out []float64) {
	out = make([]float64, size)
	FillRandom(out)
	return
}

func FillRandomMinMax(dst []float64, min, max float64) {
	for i := range dst {
		dst[i] = min + (max-min)*rand.Float64()
	}
}

func MakeRandomMinMax(size int, min, max float64) (out []float64) {
	out = make([]float64, size)
	FillRandomMinMax(out, min, max)
	return
}

func Dot(a, b []float64) (out float64) {
	for i, aV := range a {
		out += aV * b[i]
	}
	return
}

func MulTo(dst []float64, f float64) {
	for i, v := range dst {
		dst[i] = v * f
	}
}

func Mul(src []float64, f float64) (out []float64) {
	out = Copy(src)
	MulTo(out, f)
	return
}

func AddTo(dst, src []float64) {
	for i, v := range src {
		dst[i] += v
	}
}

func Add(src, vec []float64) (out []float64) {
	out = Copy(src)
	AddTo(out, vec)
	return
}

func AddScalarTo(dst []float64, f float64) {
	for i, v := range dst {
		dst[i] = v + f
	}
}

func AddScalar(src []float64, f float64) (out []float64) {
	out = Copy(src)
	AddScalarTo(out, f)
	return
}

func GetMinMaxValues(data []float64) (min, max float64) {
	for i := 0; i < len(data); i++ {
		if i == 0 || min > data[i] {
			min = data[i]
		}
		if i == 0 || max < data[i] {
			max = data[i]
		}
	}
	return
}

func GetMax(src []float64) (maxv float64, maxi int) {
	maxv, maxi = 0.0, 0
	for i, v := range src {
		if i == 0 || maxv < v {
			maxv = v
			maxi = i
		}
	}
	return
}

func ExpTo(src []float64) {
	max, _ := GetMax(src)
	for i, v := range src {
		src[i] = math.Exp(v - max)
	}
}

func LogTo(src []float64) {
	for i, v := range src {
		src[i] = math.Log(v)
	}
}

func TanhTo(src []float64) {
	for i, v := range src {
		src[i] = math.Tanh(v)
	}
}

func SigmoidTo(src []float64) {
	for i, v := range src {
		src[i] = 1 / (1 + math.Exp(-v))
	}
}

func ReluTo(src []float64) {
	for i, v := range src {
		if v < 0 {
			src[i] = 0
		}
	}
}

func Sum(src []float64) (out float64) {
	for _, v := range src {
		out += v
	}
	return
}

func CumulativeSum(src []float64) []float64 {
	res := make([]float64, len(src))
	copy(res, src)

	for i := 1; i < len(res); i++ {
		res[i] += res[i-1]
	}
	return res
}

func Multinomial(distribution []float64) (r int) {
	d := CumulativeSum(distribution)
	f := rand.Float64() * d[len(d)-1]

	for i := 0; i < len(d); i++ {
		if f <= d[i] {
			return i
		}
	}

	return len(d) - 1
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

func Conv(
	iw, ih int, inputs []float64,
	fw, fh int, filter []float64,
	channels int,
	padding int,
	stride int,
) (
	ow, oh int, output []float64,
) {
	ow, oh = CalcConvOutputSize(iw, ih, fw, fh, padding, stride)
	output = make([]float64, ow*oh)

	ConvTo(iw, ih, inputs, fw, fh, filter, ow, oh, output, channels, padding)
	return
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
	iw, ih int, image []float64,
	fw, fh int, filter []float64,
	ow, oh int, output []float64,
	channels int,
	padding int,
) {
	iw, ih, _, inputs := AddPadding(image, iw, ih, channels, padding)

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

func Conv2DTo(
	iw, ih int, image []float64,
	fw, fh int, filter2D []float64,
	ow, oh int, output []float64,
	channels int,
	padding int,
) {
	iw, ih, _, inputs := AddPadding(image, iw, ih, channels, padding)

	fHiW := fh * iw
	oHiW := oh * iw

	iSquare := iw * ih
	iCube := iSquare * channels

	oSquare := oh * ow
	oOffset := 0

	for izo := 0; izo < iCube; izo += iSquare {
		wCoord := 0
		output := output[oOffset : oOffset+oSquare]
		oOffset += oSquare

		for iyo := izo; iyo < izo+fHiW; iyo += iw {
			for ixo := iyo; ixo < iyo+fw; ixo++ {
				weight := filter2D[wCoord]
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

func ConvLayersTo(
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
			ConvLayerTo(
				ow, oh, output[outputOffset:outputOffset+outputSquare],
				iw, ih, inputs[inputsOffset:inputsOffset+inputsSquare],
				fw, fh, filter[filterOffset:filterOffset+filterSquare],
				padding,
			)

			outputOffset += outputSquare
			filterOffset += filterSquare
		}

		inputsOffset += inputsSquare
	}
}

func ConvLayerTo(
	ow, oh int, output []float64,
	iw, ih int, inputs []float64,
	fw, fh int, filter []float64,
	padding int,
) {
	iw, ih, _, inputs = AddPadding(inputs, iw, ih, 1, padding)

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
