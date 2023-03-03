package data

import (
	"math"
	"math/rand"
)

func Fill(dst []float64, v float64) {
	for i := range dst {
		dst[i] = v
	}
}

func MakeRandom(size int) (out []float64) {
	//min, max := -0.7, 0.7
	out = make([]float64, size)
	for i := range out {
		//out[i] = min + (max-min)*rand.Float64()
		out[i] = rand.Float64()
	}
	return
}

func FillRandom(dst []float64) {
	for i := range dst {
		dst[i] = rand.Float64()
	}
}

func CopyWithData(src []float64) (dst []float64) {
	dst = make([]float64, len(src))
	copy(dst, src)
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
	out = CopyWithData(src)
	MulTo(out, f)
	return
}

func AddTo(dst, src []float64) {
	for i, v := range src {
		dst[i] += v
	}
}

func Add(src, vec []float64) (out []float64) {
	out = CopyWithData(src)
	AddTo(out, vec)
	return
}

func AddScalarTo(dst []float64, f float64) {
	for i, v := range dst {
		dst[i] = v + f
	}
}

func AddScalar(src []float64, f float64) (out []float64) {
	out = CopyWithData(src)
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

func Exp(src []float64) (out []float64) {
	maxv, _ := GetMax(src)

	out = CopyWithData(src)
	for i, v := range out {
		out[i] = math.Exp(v - maxv)
	}
	return
}

func Log(src []float64) (out []float64) {
	out = CopyWithData(src)
	for i, v := range out {
		out[i] = math.Log(v)
	}
	return
}

func Tanh(src []float64) (out []float64) {
	out = CopyWithData(src)
	for i, v := range out {
		out[i] = math.Tanh(v)
	}
	return
}

func Sigmoid(src []float64) (out []float64) {
	out = CopyWithData(src)
	for i, v := range out {
		out[i] = 1 / (1 + math.Exp(-v))
	}
	return out
}

func Relu(src []float64) (out []float64) {
	out = CopyWithData(src)
	for i, v := range out {
		if v < 0 {
			out[i] = 0
		}
	}
	return
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
