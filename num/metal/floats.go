package metal

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	blas "github.com/atkhx/nnet/veclib/blas32"
	vdsp "github.com/atkhx/nnet/veclib/vdsp32"
	vforce "github.com/atkhx/nnet/veclib/vforce32"
)

func NewFloat32s(size int) Float32s {
	return make(Float32s, size)
}

func NewRandFloat32s(size int) Float32s {
	result := NewFloat32s(size)
	result.Rand()
	return result
}

func NewRandNormFloat32s(size int) Float32s {
	result := NewFloat32s(size)
	result.RandNorm()
	return result
}

func NewRandNormWeightedFloat32s(size int, w float32) Float32s {
	result := NewFloat32s(size)
	result.RandNormWeighted(w)
	return result
}

type Float32s []float32

func (f Float32s) Rand() {
	for i := range f {
		f[i] = rand.Float32()
	}
}

func (f Float32s) RandNorm() {
	for i := range f {
		f[i] = float32(rand.NormFloat64())
	}
}

func (f Float32s) RandNormWeighted(w float32) {
	for i := range f {
		f[i] = float32(rand.NormFloat64()) * w
	}
}

func (f Float32s) Fill(v float32) {
	vdsp.VfillDAll(v, f)
	//for i := range f {
	//	f[i] = v
	//}
}

func (f Float32s) Ones() {
	vdsp.VfillDAllOnes(f)
	//vdsp.VfillDAll(1., f)
	//for i := range f {
	//	f[i] = 1
	//}
}

func (f Float32s) Zero() {
	vdsp.VclrDAll(f)
	//for i := range f {
	//	f[i] = 0
	//}
}

func (f Float32s) CopyFrom(src Float32s) {
	blas.Copy(len(f), src, 1, f, 1)
}

func (f Float32s) Copy() Float32s {
	res := make(Float32s, len(f))
	res.CopyFrom(f)
	return res
}

func (f Float32s) CopyZero() Float32s {
	return make(Float32s, len(f))
}

func (f Float32s) Add(b Float32s) {
	vdsp.VaddD(f, b, f, len(f))
	//vectorAddWeighted(b, f, 1)
	//for i, v := range f {
	//	f[i] = v + b[i]
	//}
}

func (f Float32s) Sub(b Float32s) {
	blas.Axpy(len(f), -1, b, 1, f, 1)

	//veclib.VectorAddWeighted(b, f, -1)
	//for i, v := range f {
	//	f[i] = v - b[i]
	//}
}

func (f Float32s) Mul(b Float32s) {
	vdsp.VmulD(f, b, f, len(f))
	//for i, v := range f {
	//	f[i] = v * b[i]
	//}
}

func (f Float32s) MulTo(dst, b Float32s) {
	vdsp.VmulD(f, b, dst, len(f))
	//for i, v := range f {
	//	f[i] = v * b[i]
	//}
}

func (f Float32s) AddScalar(b float32) {
	vdsp.VsaddD(f, 1, f, 1, b, len(f))
	//for i := range f {
	//	f[i] += b
	//}
}

func (f Float32s) AddWeighted(b Float32s, w float32) {
	blas.Axpy(len(f), w, b, 1, f, 1)
	//for i, v := range b {
	//	f[i] += v * w
	//}
}

func (f Float32s) MulScalar(b float32) {
	blas.Scal(len(f), b, f, 1)
	//for i := range f {
	//	f[i] *= b
	//}
}

func (f Float32s) Max() float32 {
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

func (f Float32s) Pow2() {
	vdsp.VmulD(f, f, f, len(f))
	//vforce.VvsqrtD(f, f, len(f))
	//for i, v := range f {
	//	f[i] *= v
	//}
}

func (f Float32s) Sqrt() {
	vforce.VvsqrtD(f, f, len(f))
}

func (f Float32s) SqrtTo(dst Float32s) {
	vforce.VvsqrtD(f, dst, len(f))
}

func (f Float32s) AddKXDivYTo(dst, yData Float32s) {
	for i, y := range yData {
		dst[i] += f[i] * 0.5 / y
	}
}

func (f Float32s) MaxKeyVal() (maxI int, maxV float32) {
	for i, v := range f {
		if i == 0 || maxV < v {
			maxV = v
			maxI = i
		}
	}
	return
}

func (f Float32s) MaxIndex() (maxIndex int) {
	var max = f[0]
	for i, v := range f {
		if max < v {
			max = v
			maxIndex = i
		}
	}
	return
}

func (f Float32s) Min() (min float32) {
	min = f[0]
	for _, v := range f {
		if min > v {
			min = v
		}
	}
	return
}

func (f Float32s) Exp() {
	vforce.VvexpD(f, f, len(f))
	//for i, v := range f {
	//	f[i] = math.Exp(v)
	//}
}

func (f Float32s) Mean() (sum float32) {
	return f.Sum() / float32(len(f))
}

func (f Float32s) Std() float32 {
	mean := f.Mean()

	out := 0.0
	for _, v := range f {
		out += math.Pow(float64(v-mean), 2)
	}
	out /= float64(len(f))
	out = math.Sqrt(out)

	return float32(out)
}

func (f Float32s) Sum() (sum float32) {
	return vdsp.SveD(f, 1, len(f))
	//for _, v := range f {
	//	sum += v
	//}
	//return
}

func (f Float32s) Softmax() {
	f.AddScalar(-f.Max())
	f.Exp()
	f.MulScalar(1. / f.Sum())
}

func (f Float32s) SoftmaxTo(out Float32s) {
	out.CopyFrom(f)
	out.Softmax()
}

func (f Float32s) CumulativeSum() {
	for i := 1; i < len(f); i++ {
		f[i] += f[i-1]
	}
}

func (f Float32s) Multinomial() (r int) {
	// f - distribution
	v := rand.Float32() * f[len(f)-1] //nolint:gosec
	for i, w := range f {
		if v <= w {
			return i
		}
	}
	return len(f) - 1
}

func (f Float32s) ToInt() (r []int) {
	r = make([]int, len(f))
	for i, v := range f {
		r[i] = int(v)
	}
	return
}

func (f Float32s) ToBytes() (r []byte) {
	r = make([]byte, len(f))
	for i, v := range f {
		r[i] = byte(v)
	}
	return
}

func (f Float32s) FromBytes(b []byte) {
	for i, v := range b {
		f[i] = float32(v)
	}
}

func (f Float32s) String(dims Dims) string {
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
