package data

import (
	"fmt"
	"math"
	"math/rand"
)

const (
	ReLuGain = 1.4142135624
	TanhGain = 1.6666666667
)

func Round(v, k float64) float64 {
	return math.Round(v*k) / k
}

func RoundFloats(src []float64, k float64) []float64 {
	out := Copy(src)
	for i, v := range out {
		out[i] = math.Round(v*k) / k
	}
	return out
}

func RoundTo(floats []float64, k float64) {
	for i, v := range floats {
		floats[i] = math.Round(v*k) / k
	}
}

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
	//rand.Seed(time.Now().UnixNano())
	for i := range dst {
		dst[i] = rand.NormFloat64()
		//dst[i] = rand.Float64()
	}

	//fmt.Println(dst)
	//os.Exit(1)
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

func DivTo(dst []float64, f float64) {
	for i, v := range dst {
		dst[i] = v / f
	}
}

func Div(src []float64, f float64) (out []float64) {
	out = Copy(src)
	DivTo(out, f)
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
	//max, _ := GetMax(src)
	for i, v := range src {
		//src[i] = math.Exp(v - max)
		src[i] = math.Exp(v)
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

func Convolve(
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

	ConvolveTo(ow, oh, output, iw, ih, inputs, fw, fh, filter, channels, padding)
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

func ConvolveTo(
	ow, oh int, output []float64,
	iw, ih int, inputs []float64,
	fw, fh int, filter []float64,
	channels int,
	padding int,
) {
	ConvolveTo2(ow, oh, output, iw, ih, inputs, fw, fh, filter, channels, padding)
	//iw, ih, _, inputs = AddPadding(inputs, iw, ih, channels, padding)
	//ConvolvePaddedTo(ow, oh, output, iw, ih, inputs, fw, fh, filter, channels)
}

func ConvolvePaddedTo(
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

func ConvolveTo2(
	ow, oh int, output []float64,
	iw, ih int, inputs []float64,
	fw, fh int, filter []float64,
	channels int,
	padding int,
) {
	iSquare := iw * ih
	iCube := iSquare * channels

	wCoord := 0

	for izFrom := 0; izFrom < iCube; izFrom += iSquare {
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

// Convolve2dBatchTo convolve each 2d-input by each 2d-filter and store separate convolution result in output
func Convolve2dBatchTo(
	ow, oh int, output []float64,
	iw, ih, ic int, inputs []float64,
	fw, fh, fc int, filter []float64,
	padding int,
) {
	//iw, ih, _, inputs = AddPadding(inputs, iw, ih, ic*fc, padding)

	outputSquare := ow * oh
	inputsSquare := iw * ih
	filterSquare := fw * fh

	outputOffset := 0
	inputsOffset := 0

	for ii := 0; ii < ic; ii++ {
		filterOffset := 0
		for fi := 0; fi < fc; fi++ {
			ConvolveTo2(
				ow, oh, output[outputOffset:outputOffset+outputSquare],
				iw, ih, inputs[inputsOffset:inputsOffset+inputsSquare],
				fw, fh, filter[filterOffset:filterOffset+filterSquare],
				1,
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
