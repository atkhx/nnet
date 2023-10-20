package native

import (
	"math"
	"math/rand"
	"time"
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
	for i := range f {
		f[i] = v
	}
}

func (f Float32s) Ones() {
	for i := range f {
		f[i] = 1
	}
}

func (f Float32s) Zero() {
	for i := range f {
		f[i] = 0
	}
}

func (f Float32s) CopyFrom(src Float32s) {
	copy(f, src)
}

func (f Float32s) Copy() Float32s {
	res := make(Float32s, len(f))
	copy(res, f)
	return res
}

func (f Float32s) CopyZero() Float32s {
	return make(Float32s, len(f))
}

func (f Float32s) Add(b Float32s) {
	for i, v := range b {
		f[i] += v
	}
}

func (f Float32s) AddTo(dst, b Float32s) {
	for i, v := range b {
		dst[i] = f[i] + v
	}
}

func (f Float32s) Sub(b Float32s) {
	for i, v := range b {
		f[i] -= v
	}
}

func (f Float32s) Mul(b Float32s) {
	for i, v := range b {
		f[i] *= v
	}
}

func (f Float32s) AddWeighted(b Float32s, w float32) {
	for i, v := range b {
		f[i] += w * v
	}
}

func (f Float32s) AddScalar(b float32) {
	for i := range f {
		f[i] += b
	}
}

func (f Float32s) AddScalarTo(dst Float32s, b float32) {
	for i, v := range f {
		dst[i] = v + b
	}
}

func (f Float32s) MulScalar(b float32) {
	for i := range f {
		f[i] *= b
	}
}

func (f Float32s) MulScalarTo(dst Float32s, b float32) {
	for i, v := range f {
		dst[i] = v * b
	}
}

func (f Float32s) Max() (max float32) {
	max = f[0]
	for _, v := range f {
		if max < v {
			max = v
		}
	}
	return
}

func (f Float32s) Pow2() {
	for i, v := range f {
		f[i] *= v
	}
}

func (f Float32s) Sqrt() {
	for i, v := range f {
		f[i] = float32(math.Sqrt(float64(v)))
	}
}

func (f Float32s) SqrtTo(dst Float32s) {
	for i, v := range f {
		dst[i] = float32(math.Sqrt(float64(v)))
	}
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
	for i, v := range f {
		f[i] = float32(math.Exp(float64(v)))
	}
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
	for _, v := range f {
		sum += v
	}
	return
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

func (f Float32s) MatrixMultiply2DTo(bData, dst Float32s, aW int, alpha, beta float32) {
	bW := len(bData) / aW
	aH := len(f) / aW

	dst.MulScalar(beta)

	for outRow := 0; outRow < aH; outRow++ {
		dst := dst[outRow*bW : outRow*bW+bW]
		f := f[outRow*aW : outRow*aW+aW]
		for bRow, v := range f {
			if v != 0 {
				v *= alpha
				bData := bData[bRow*bW : (bRow+1)*bW]
				for bCol, cv := range dst {
					dst[bCol] = cv + v*bData[bCol]
				}
			}
		}
	}
}

func (f Float32s) MatrixMultiply2DTo2(bData, dst Float32s, aW int, alpha, beta float32) {
	bW := len(bData) / aW
	aH := len(f) / aW

	offset := 0
	for outRow := 0; outRow < aH; outRow++ { // aRow
		for outCol := 0; outCol < bW; outCol++ { // bCol
			var res float32
			for aCol := 0; aCol < aW; aCol++ {
				if v := f[outRow*aW+aCol]; v != 0 {
					res += v * bData[aCol*bW+outCol]
				}
			}
			dst[offset] = beta*dst[offset] + alpha*res
			offset++
		}
	}
}

func (f Float32s) MatrixMultiply2DAonTransposedBTo(bData, dst Float32s, aW int, alpha, beta float32) {
	bH := len(bData) / aW
	aH := len(f) / aW

	offset := 0
	for outRow := 0; outRow < aH; outRow++ { // aRow
		for outCol := 0; outCol < bH; outCol++ { // now it's bRow
			var res float32
			for aCol := 0; aCol < aW; aCol++ {
				if v := f[outRow*aW+aCol]; v != 0 {
					res += v * bData[outCol*aW+aCol]
				}
			}
			dst[offset] = beta*dst[offset] + alpha*res
			offset++
		}
	}
}

func (f Float32s) MatrixMultiply2DTransposedAonBTo(bData, dst Float32s, aH int, alpha, beta float32) {
	bW := len(bData) / aH
	aW := len(f) / aH

	offset := 0
	for outRow := 0; outRow < aW; outRow++ {
		for outCol := 0; outCol < bW; outCol++ { // bCol
			var res float32
			for aRow := 0; aRow < aH; aRow++ {
				if v := f[aRow*aW+outRow]; v != 0 {
					res += v * bData[aRow*bW+outCol]
				}
			}
			dst[offset] = beta*dst[offset] + alpha*res
			offset++
		}
	}
}
