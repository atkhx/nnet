package num

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/atkhx/nnet/veclib/blas"
	"github.com/atkhx/nnet/veclib/vdsp"
	"github.com/atkhx/nnet/veclib/vforce"
)

func NewFloat64s(size int) Float64s {
	return make(Float64s, size)
}

func NewRandFloat64s(size int) Float64s {
	result := NewFloat64s(size)
	result.Rand()
	return result
}

func NewRandNormFloat64s(size int) Float64s {
	result := NewFloat64s(size)
	result.RandNorm()
	return result
}

func NewRandNormWeightedFloat64s(size int, w float64) Float64s {
	result := NewFloat64s(size)
	result.RandNormWeighted(w)
	return result
}

type Float64s []float64

func (f Float64s) Rand() {
	for i := range f {
		f[i] = rand.Float64()
	}
}

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
	vdsp.VfillDAll(v, f)
	//for i := range f {
	//	f[i] = v
	//}
}

func (f Float64s) Ones() {
	vdsp.VfillDAllOnes(f)
	//vdsp.VfillDAll(1., f)
	//for i := range f {
	//	f[i] = 1
	//}
}

func (f Float64s) Zero() {
	vdsp.VclrDAll(f)
	//for i := range f {
	//	f[i] = 0
	//}
}

func (f Float64s) CopyFrom(src Float64s) {
	blas.DCopy(len(f), src, 1, f, 1)
}

func (f Float64s) Copy() Float64s {
	res := make(Float64s, len(f))
	res.CopyFrom(f)
	return res
}

func (f Float64s) CopyZero() Float64s {
	return make(Float64s, len(f))
}

func (f Float64s) Add(b Float64s) {
	vdsp.VaddD(f, b, f, len(f))
	//vectorAddWeighted(b, f, 1)
	//for i, v := range f {
	//	f[i] = v + b[i]
	//}
}

func (f Float64s) Sub(b Float64s) {
	blas.Daxpy(len(f), -1, b, 1, f, 1)

	//veclib.VectorAddWeighted(b, f, -1)
	//for i, v := range f {
	//	f[i] = v - b[i]
	//}
}

func (f Float64s) Mul(b Float64s) {
	vdsp.VmulD(f, b, f, len(f))
	//for i, v := range f {
	//	f[i] = v * b[i]
	//}
}

func (f Float64s) MulTo(dst, b Float64s) {
	vdsp.VmulD(f, b, dst, len(f))
	//for i, v := range f {
	//	f[i] = v * b[i]
	//}
}

func (f Float64s) AddScalar(b float64) {
	vdsp.VsaddD(f, 1, f, 1, b, len(f))
	//for i := range f {
	//	f[i] += b
	//}
}

func (f Float64s) AddWeighted(b Float64s, w float64) {
	blas.Daxpy(len(f), w, b, 1, f, 1)
	//for i, v := range b {
	//	f[i] += v * w
	//}
}

func (f Float64s) MulScalar(b float64) {
	blas.Dscal(len(f), b, f, 1)
	//for i := range f {
	//	f[i] *= b
	//}
}

func (f Float64s) Max() float64 {
	return vdsp.MaxvD(f, 1, len(f))
	//
	//max = f[0]
	//for _, v := range f {
	//	if max < v {
	//		max = v
	//	}
	//}
	//return
}

func (f Float64s) Pow2() {
	vdsp.VmulD(f, f, f, len(f))
	//vforce.VvsqrtD(f, f, len(f))
	//for i, v := range f {
	//	f[i] *= v
	//}
}

func (f Float64s) Sqrt() {
	vforce.VvsqrtD(f, f, len(f))
}

func (f Float64s) SqrtTo(dst Float64s) {
	vforce.VvsqrtD(f, dst, len(f))
}

func (f Float64s) MaxKeyVal() (maxI int, maxV float64) {
	for i, v := range f {
		if i == 0 || maxV < v {
			maxV = v
			maxI = i
		}
	}
	return
}

func (f Float64s) MaxIndex() (maxIndex int) {
	var max = f[0]
	for i, v := range f {
		if max < v {
			max = v
			maxIndex = i
		}
	}
	return
}

func (f Float64s) Min() (min float64) {
	min = f[0]
	for _, v := range f {
		if min > v {
			min = v
		}
	}
	return
}

func (f Float64s) Exp() {
	vforce.VvexpD(f, f, len(f))
	//for i, v := range f {
	//	f[i] = math.Exp(v)
	//}
}

func (f Float64s) Mean() (sum float64) {
	return f.Sum() / float64(len(f))
}

func (f Float64s) Std() float64 {
	mean := f.Mean()

	out := 0.0
	for _, v := range f {
		out += math.Pow(v-mean, 2)
	}
	out /= float64(len(f))
	out = math.Sqrt(out)

	return out
}

func (f Float64s) Sum() (sum float64) {
	return vdsp.SveD(f, 1, len(f))
	//for _, v := range f {
	//	sum += v
	//}
	//return
}

func (f Float64s) Softmax() {
	f.AddScalar(-f.Max())
	f.Exp()
	f.MulScalar(1. / f.Sum())
}

func (f Float64s) SoftmaxTo(out Float64s) {
	out.CopyFrom(f)
	out.Softmax()
}

func (f Float64s) CumulativeSum() {
	for i := 1; i < len(f); i++ {
		f[i] += f[i-1]
	}
}

func (f Float64s) Multinomial() (r int) {
	// f - distribution
	v := rand.Float64() * f[len(f)-1] //nolint:gosec
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

func (f Float64s) String(dims Dims) string {
	var result string
	var offset int

	result += strings.Repeat("-", 40) + "\n"
	// result += "[\n"
	for z := 0; z < dims.D; z++ {
		if z > 0 {
			result += "\n"
		}

		// result += "  [\n"
		for y := 0; y < dims.H; y++ {
			if y > 0 {
				result += "\n"
			}
			result += "    "
			for x := 0; x < dims.W; x++ {
				v := f[offset]
				if v >= 0 {
					result += " "
				}

				result += fmt.Sprintf("%.7f ", v)
				offset++
			}
		}
		// result += "\n"
		// result += "\n  ]"
	}
	// result += "\n]"

	return result
}
