package num

import "math/rand"

const (
	ReLuGain    = 1.4142135624
	TanhGain    = 1.6666666667
	SigmoidGain = 1.0
	LinearGain  = 1.0
)

func Dot(a, b []float64) (r float64) {
	for i, aV := range a {
		r += aV * b[i]
	}
	return
}

type Float64s []float64

func (f Float64s) RandNormWeighted(w float64) {
	for i := range f {
		f[i] = rand.NormFloat64() * w
	}
}

func (f Float64s) Fill(v float64) {
	for i := range f {
		f[i] = v
	}
}

func (f Float64s) Add(b Float64s) {
	for i, v := range f {
		f[i] = v + b[i]
	}
}

func (f Float64s) AddWeighted(b Float64s, w float64) {
	for i, v := range f {
		f[i] = v + b[i]*w
	}
}
