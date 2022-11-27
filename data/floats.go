package data

func Fill(dst []float64, value float64) {
	for i := range dst {
		dst[i] = value
	}
}

func Dot(sliceA, sliceB []float64) (dot float64) {
	for i, v := range sliceA {
		dot += v * sliceB[i]
	}
	return
}

func MultiplyAndAddTo(dst, src []float64, k float64) {
	for ic, iv := range src {
		dst[ic] += iv * k
	}
}

func SumElements(src []float64) (sum float64) {
	for _, v := range src {
		sum += v
	}
	return
}

func SumSlicesTo(dst []float64, src ...[]float64) {
	for _, srcSlice := range src {
		for j, v := range srcSlice {
			dst[j] += v
		}
	}
}

func SumSlices(src ...[]float64) (dst []float64) {
	dst = make([]float64, len(src[0]))
	for _, srcSlice := range src {
		for j, v := range srcSlice {
			dst[j] += v
		}
	}
	return
}
