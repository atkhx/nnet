package floats

import (
	"math"
	"math/rand"
)

func Round(floats []float64, k float64) {
	for i, v := range floats {
		floats[i] = math.Round(v*k) / k
	}
}

func Fill(dst []float64, value float64) {
	for i := range dst {
		dst[i] = value
	}
}

func FillRandom(dst []float64, min, max float64) {
	for i := range dst {
		//nolint:gosec
		dst[i] = min + (max-min)*rand.Float64()
	}
}

func AddTo(dst []float64, src ...[]float64) {
	for _, items := range src {
		for j, v := range items {
			dst[j] += v
		}
	}
}

func Add(src ...[]float64) (dst []float64) {
	for i, items := range src {
		if i == 0 {
			dst = make([]float64, len(items))
			copy(dst, items)
			continue
		}

		for j, v := range items {
			dst[j] += v
		}
	}
	return
}

func GetMaxIndex(data []float64) (index int) {
	var value float64
	for i := 0; i < len(data); i++ {
		if i == 0 || value < data[i] {
			value = data[i]
			index = i
		}
	}
	return
}

func GetMaxValue(data []float64) (max float64) {
	for i := 0; i < len(data); i++ {
		if i == 0 || max < data[i] {
			max = data[i]
		}
	}
	return
}

func GetMinValue(data []float64) (min float64) {
	for i := 0; i < len(data); i++ {
		if i == 0 || min > data[i] {
			min = data[i]
		}
	}
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

func GetMinMaxValuesInRange(data []float64, from, to int) (float64, float64) {
	return GetMinMaxValues(data[from:to])
}

func SumElements(src []float64) (sum float64) {
	for _, v := range src {
		sum += v
	}
	return
}

func MultiplyAndAdd(src []float64, k float64) (dst []float64) {
	dst = make([]float64, len(src))
	MultiplyAndAddTo(dst, src, k)

	return dst
}

func CumulativeSum(src []float64) []float64 {
	res := make([]float64, len(src))
	copy(res, src)

	for i := 1; i < len(res); i++ {
		res[i] += res[i-1]
	}
	return res
}

//nolint:gosec
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
