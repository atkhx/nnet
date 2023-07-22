package num

import (
	"C"
)

func max[T int | int64 | float64](a, b T) T {
	if a > b {
		return a
	}
	return b
}

func pointer[T any](v T) *T {
	return &v
}

func mm_tr_lower(aW int, a, b, c Float64s) {
	oW := len(b) / aW

	for y := 0; y < aW; y++ {
		for x, v := range a[y*aW : y*aW+y+1] {
			if v != 0 {
				axpyUnitary(v, b[x*oW:x*oW+oW], c[y*oW:y*oW+oW])
			}
		}
	}
}

func axpyUnitary(alpha float64, x, y []float64) {
	for i, v := range x {
		y[i] += alpha * v
	}
}

func transpose(aW, aH int, aData Float64s) Float64s {
	oData := aData.CopyZero()
	transposeTo(aW, aH, aData, oData)
	return oData
}

func transposeTo(aW, aH int, aData, oData Float64s) {
	WH := aW * aH
	for d := 0; d < len(aData); d += WH {
		for y := 0; y < aH; y++ {
			for x := 0; x < aW; x++ {
				oData[d+x*aH+y] = aData[d+y*aW+x]
			}
		}
	}
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

func dot(iData, fData Float64s) (v float64) {
	for i, iV := range iData {
		v += iV * fData[i]
	}
	return v
}
