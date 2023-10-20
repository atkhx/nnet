package metal

import (
	"math"
	"math/rand"
	"time"

	blas "github.com/atkhx/nnet/veclib/blas32"
	vdsp "github.com/atkhx/nnet/veclib/vdsp32"
	vforce "github.com/atkhx/nnet/veclib/vforce32"
)

var randGenerator = rand.New(rand.NewSource(time.Now().UnixNano()))

func NewFloat32s(size int) Float32s {
	return make(Float32s, size)
}

type Float32s []float32

func (f Float32s) Rand() {
	for i := range f {
		f[i] = randGenerator.Float32()
	}
}

func (f Float32s) RandNorm() {
	for i := range f {
		f[i] = float32(randGenerator.NormFloat64())
	}
}

func (f Float32s) RandNormWeighted(w float32) {
	for i := range f {
		f[i] = float32(randGenerator.NormFloat64()) * w
	}
}

func (f Float32s) Fill(v float32) {
	vdsp.VfillDAll(v, f)
}

func (f Float32s) Ones() {
	vdsp.VfillDAllOnes(f)
}

func (f Float32s) Zero() {
	vdsp.VclrDAll(f)
}

func (f Float32s) CopyFrom(src Float32s) {
	blas.Copy(len(f), src, 1, f, 1)
}

func (f Float32s) Copy() Float32s {
	res := make(Float32s, len(f))
	blas.Copy(len(f), f, 1, res, 1)
	return res
}

func (f Float32s) CopyZero() Float32s {
	return make(Float32s, len(f))
}

func (f Float32s) Add(b Float32s) {
	vdsp.VaddD(f, b, f, len(f))
}

func (f Float32s) AddTo(dst, b Float32s) {
	vdsp.VaddD(f, b, dst, len(f))
}

func (f Float32s) Sub(b Float32s) {
	blas.Axpy(len(f), -1, b, 1, f, 1)
}

func (f Float32s) Mul(b Float32s) {
	vdsp.VmulD(f, b, f, len(f))
}

func (f Float32s) AddWeighted(b Float32s, w float32) {
	blas.Axpy(len(f), w, b, 1, f, 1)
}

func (f Float32s) AddScalar(b float32) {
	vdsp.VsaddD(f, 1, f, 1, b, len(f))
}

func (f Float32s) AddScalarTo(dst Float32s, b float32) {
	vdsp.VsaddD(f, 1, dst, 1, b, len(f))
}

func (f Float32s) MulScalar(b float32) {
	blas.Scal(len(f), b, f, 1)
}

func (f Float32s) MulScalarTo(dst Float32s, b float32) {
	dst.CopyFrom(f)
	dst.MulScalar(b)
}

func (f Float32s) Max() (max float32) {
	return vdsp.MaxvD(f, 1, len(f))
}

func (f Float32s) Pow2() {
	vdsp.VmulD(f, f, f, len(f))
}

func (f Float32s) Sqrt() {
	vforce.VvsqrtD(f, f, len(f))
}

func (f Float32s) SqrtTo(dst Float32s) {
	vforce.VvsqrtD(f, dst, len(f))
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
	//	f[i] = float32(math.Exp(float64(v)))
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

func (f Float32s) Variance(mean float32) float32 {
	out := float32(0.0)
	for _, v := range f {
		out += (v - mean) * (v - mean)
	}
	return out / float32(len(f)-1)
}

func (f Float32s) StdDev(mean float32) float32 {
	return float32(math.Sqrt(float64(f.Variance(mean))))
}

func (f Float32s) Sum() (sum float32) {
	return vdsp.SveD(f, 1, len(f))
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
	v := randGenerator.Float32() * f[len(f)-1]
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

func (f Float32s) TransposeTo(dst Float32s, aW, aH int) {
	for d := 0; d < len(f); d += aW * aH {
		for y := 0; y < aH; y++ {
			for x := 0; x < aW; x++ {
				dst[d+x*aH+y] = f[d+y*aW+x]
			}
		}
	}
}

func (f Float32s) TransposeAndAddTo(dst Float32s, aW, aH int) {
	for d := 0; d < len(f); d += aW * aH {
		for y := 0; y < aH; y++ {
			for x := 0; x < aW; x++ {
				dst[d+x*aH+y] += f[d+y*aW+x]
			}
		}
	}
}
