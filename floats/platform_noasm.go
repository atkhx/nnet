//go:build !amd64 || noasm

package floats

func Dot(sliceA, sliceB []float64) (dot float64) {
	for i, v := range sliceA {
		dot += v * sliceB[i]
	}
	return
}

func MultiplyTo(dst, src []float64, k float64) {
	for ic, iv := range src {
		dst[ic] = iv * k
	}
}

func MultiplyAndAddTo(dst, src []float64, k float64) {
	for ic, iv := range src {
		dst[ic] += iv * k
	}
}
