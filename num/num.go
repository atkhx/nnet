package num

func Dot(a, b []float64) (r float64) {
	for i, aV := range a {
		r += aV * b[i]
	}
	return
}
