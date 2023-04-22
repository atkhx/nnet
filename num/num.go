package num

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

func Dot(a, b []float64) (r float64) {
	for i, aV := range a {
		r += aV * b[i]
	}
	return
}

type Float64s []float64

func (f Float64s) RandNorm() {
	for i := range f {
		f[i] = rand.NormFloat64()
	}
}

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

func (f Float64s) Copy() Float64s {
	res := make(Float64s, len(f))
	copy(res, f)
	return res
}

func (f Float64s) Add(b Float64s) {
	for i, v := range f {
		f[i] = v + b[i]
	}
}

func (f Float64s) AddScalar(b float64) {
	for i, v := range f {
		f[i] = v + b
	}
}

func (f Float64s) AddWeighted(b Float64s, w float64) {
	for i, v := range f {
		f[i] = v + b[i]*w
	}
}

func (f Float64s) MulScalar(b float64) {
	for i, v := range f {
		f[i] = v * b
	}
}

func (f Float64s) Max() (max float64) {
	for i, v := range f {
		if i == 0 || max < v {
			max = v
		}
	}
	return
}

func (f Float64s) Min() (min float64) {
	for i, v := range f {
		if i == 0 || min > v {
			min = v
		}
	}
	return
}

func (f Float64s) Exp() {
	max := f.Max()
	for i, v := range f {
		f[i] = math.Exp(v - max)
	}
}

func (f Float64s) Sum() (sum float64) {
	for _, v := range f {
		sum += v
	}
	return
}

func (f Float64s) Softmax() {
	f.Exp()

	sum := f.Sum()
	for i := range f {
		f[i] /= sum
	}
}

func (f Float64s) CumulativeSum() {
	for i := 1; i < len(f); i++ {
		f[i] += f[i-1]
	}
}

func (f Float64s) Multinomial() (r int) {
	// f - distribution
	v := rand.Float64() * f[len(f)-1]
	for i, w := range f {
		if v <= w {
			return i
		}
	}
	return len(f) - 1
}

func (f Float64s) ToInt() (r []int) {
	r = make([]int, len(f))
	for i, v := range f {
		r[i] = int(v)
	}
	return
}

func (f Float64s) ToBytes() (r []byte) {
	r = make([]byte, len(f))
	for i, v := range f {
		r[i] = byte(v)
	}
	return
}

func (f Float64s) FromBytes(b []byte) {
	for i, v := range b {
		f[i] = float64(v)
	}
}
