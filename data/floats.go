package data

import (
	"math"
	"math/rand"
)

const (
	ReLuGain    = 1.4142135624
	TanhGain    = 1.6666666667
	SigmoidGain = 1.0
	LinearGain  = 1.0
)

func Round(v, k float64) float64 {
	return math.Round(v*k) / k
}

func RoundFloats(src []float64, k float64) []float64 {
	out := Copy(src)
	RoundFloatsTo(out, k)
	return out
}

func RoundFloatsTo(floats []float64, k float64) {
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
	for i := range dst {
		dst[i] = rand.NormFloat64()
	}
}

func MakeRandom(size int) (out []float64) {
	out = make([]float64, size)
	FillRandom(out)
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

func PowTo(src []float64, pow float64) {
	for i, v := range src {
		src[i] = math.Pow(v, pow)
	}
}

func SqrtTo(src []float64) {
	for i, v := range src {
		src[i] = math.Sqrt(v)
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
