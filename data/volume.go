package data

import (
	"fmt"
	"math"
	"math/rand"
)

func NewVolume(w, h, d int) *Volume {
	return &Volume{W: w, H: h, D: d, Data: make([]float64, w*h*d)}
}

func WrapVolume(w, h, d int, data []float64) *Volume {
	return &Volume{W: w, H: h, D: d, Data: data}
}

type Volume struct {
	W, H, D int

	Data []float64
}

func (obj *Volume) IsDimensionsEqual(t *Data) bool {
	return obj.W == t.Data.W && obj.H == t.Data.H && obj.D == t.Data.D
}

func (obj *Volume) GetDims() []int {
	return []int{obj.W, obj.H, obj.D}
}

func (obj *Volume) GetLen() int {
	return len(obj.Data)
}

func (obj *Volume) GetMax() (maxValue float64, maxIndex int) {
	return GetMax(obj.Data)
}

func (obj *Volume) Sum() *Volume {
	return WrapVolume(1, 1, 1, []float64{Sum(obj.Data)})
}

func (obj *Volume) SumByCols() *Volume {
	out := NewVolume(obj.W, 1, obj.D)
	obj.EachRow(func(row []float64) {
		out.AddFloats(row)
	})
	return out
}

func (obj *Volume) SumByRows() *Volume {
	out := NewVolume(1, obj.H, obj.D)
	obj.EachRowAbsNum(func(row []float64, n int) {
		out.Data[n] = Sum(row)
	})
	return out
}

func (obj *Volume) Mean() *Volume {
	return WrapVolume(1, 1, 1, []float64{Sum(obj.Data) / float64(obj.GetLen())})
}

func (obj *Volume) MeanByCols() *Volume {
	return obj.SumByCols().MulScalar(1.0 / float64(obj.H))
}

func (obj *Volume) MeanByRows() *Volume {
	return obj.SumByRows().MulScalar(1.0 / float64(obj.W))
}

func (obj *Volume) Std() *Volume {
	mean := obj.Mean().Data[0]

	out := 0.0
	for _, v := range obj.Data {
		out += math.Pow(v-mean, 2)
	}
	out /= float64(obj.GetLen() - 1)
	out = math.Sqrt(out)

	return WrapVolume(1, 1, 1, []float64{out})
}

func (obj *Volume) ColStd() (*Volume, *Volume) {
	colMean := obj.MeanByCols()
	colStd := NewVolume(colMean.W, 1, colMean.D)

	k := 1.0 / float64(obj.H)

	obj.Scan(func(x, y, z int, offset int, v float64) {
		colStd.PointAdd(x, 0, z, math.Pow(v-colMean.At(x, 0, z), 2))
	})

	colStd.Scan(func(x, _, z int, offset int, v float64) {
		colStd.Data[offset] = k * math.Sqrt(v)
	})

	return colMean, colStd
}

func (obj *Volume) Copy() *Volume {
	return WrapVolume(obj.W, obj.H, obj.D, Copy(obj.Data))
}

func (obj *Volume) Reshape(w, h, d int) *Volume {
	return WrapVolume(w, h, d, obj.Data)
}

func (obj *Volume) Transpose() *Volume {
	r := NewVolume(obj.H, obj.W, obj.D)
	r.ScanRows(func(y, z int, f []float64) {
		for x := 0; x < r.W; x++ {
			f[x] = obj.At(y, x, z)
		}
	})
	return r
}

func (obj *Volume) Rotate180() *Volume {
	out := NewVolume(obj.H, obj.W, obj.D)
	obj.ScanRows(func(y, z int, f []float64) {
		for x := 0; x < obj.W; x++ {
			out.Set(obj.W-x-1, obj.H-y-1, z, f[x])
		}
	})

	return out
}

func (obj *Volume) MatrixMultiply(b *Volume) *Volume {
	if obj.W != b.H {
		panic(fmt.Sprintf("obj.W != b.H: %d != %d", obj.W, b.H))
	}

	r := NewVolume(b.W, obj.H, obj.D)

	bT := b.Transpose()
	bT.ScanRows(func(weightIndex, _ int, bFloats []float64) {
		obj.ScanRows(func(inputIndex, z int, aFloats []float64) {
			r.Set(weightIndex, inputIndex, z, Dot(aFloats, bFloats))
		})
	})
	return r
}

func (obj *Volume) Fill(v float64) {
	for i := range obj.Data {
		obj.Data[i] = v
	}
}

func (obj *Volume) FillRandom() {
	for i := range obj.Data {
		obj.Data[i] = rand.Float64()
	}
	return
}

func (obj *Volume) FillRandomMinMax(min, max float64) {
	for i := range obj.Data {
		obj.Data[i] = min + (max-min)*rand.Float64()
	}
	return
}

func (obj *Volume) AddScalar(f float64) {
	for i, v := range obj.Data {
		obj.Data[i] = v + f
	}
}

func (obj *Volume) Add(src *Volume) {
	obj.AddFloats(src.Data)
}

func (obj *Volume) AddFloats(src []float64) {
	for i, v := range src {
		obj.Data[i] += v
	}
}

func (obj *Volume) SubScalar(f float64) {
	for i, v := range obj.Data {
		obj.Data[i] = v - f
	}
}

func (obj *Volume) Sub(src *Volume) {
	obj.SubFloats(src.Data)
}

func (obj *Volume) SubFloats(src []float64) {
	for i, v := range src {
		obj.Data[i] -= v
	}
}

func (obj *Volume) MulScalar(f float64) *Volume {
	MulTo(obj.Data, f)
	return obj
}

func (obj *Volume) Mul(src *Volume) *Volume {
	obj.MulFloats(src.Data)
	return obj
}

func (obj *Volume) MulFloats(src []float64) {
	for i, v := range src {
		obj.Data[i] *= v
	}
}

func (obj *Volume) DivScalar(f float64) *Volume {
	DivTo(obj.Data, f)
	return obj
}

func (obj *Volume) Div(src *Volume) *Volume {
	obj.DivFloats(src.Data)
	return obj
}

func (obj *Volume) DivFloats(src []float64) {
	for i, v := range src {
		obj.Data[i] /= v
	}
}

func (obj *Volume) Tanh() *Volume {
	TanhTo(obj.Data)
	return obj
}

func (obj *Volume) Sigmoid() *Volume {
	SigmoidTo(obj.Data)
	return obj
}

func (obj *Volume) Relu() *Volume {
	ReluTo(obj.Data)
	return obj
}

func (obj *Volume) Exp() *Volume {
	ExpTo(obj.Data)
	return obj
}

func (obj *Volume) Log() *Volume {
	LogTo(obj.Data)
	return obj
}

func (obj *Volume) Pow(pow float64) *Volume {
	PowTo(obj.Data, pow)
	return obj
}

func (obj *Volume) Sqrt() *Volume {
	SqrtTo(obj.Data)
	return obj
}

func (obj *Volume) Softmax() *Volume {
	exps := obj.Exp()
	sums := exps.Sum().Data[0]
	for i := range obj.Data {
		obj.Data[i] /= sums
	}
	return obj
}

func (obj *Volume) PointAdd(x, y, z int, v float64) {
	obj.Data[z*obj.W*obj.H+y*obj.W+x] += v
}

func (obj *Volume) Set(x, y, z int, v float64) {
	obj.Data[z*obj.W*obj.H+y*obj.W+x] = v
}

func (obj *Volume) At(x, y, z int) float64 {
	return obj.Data[z*obj.W*obj.H+y*obj.W+x]
}

func (obj *Volume) WrapRow(f []float64) *Volume {
	return WrapVolume(obj.W, 1, 1, f)
}

func (obj *Volume) EachRow(fn func(f []float64)) {
	for offset := 0; offset < obj.GetLen(); offset += obj.W {
		fn(obj.Data[offset : offset+obj.W])
	}
}

func (obj *Volume) EachRowVolume(fn func(f *Volume)) {
	for offset := 0; offset < obj.GetLen(); offset += obj.W {
		fn(obj.WrapRow(obj.Data[offset : offset+obj.W]))
	}
}

func (obj *Volume) EachRowAbsNum(fn func(f []float64, n int)) {
	for offset, n := 0, 0; offset < obj.GetLen(); offset, n = offset+obj.W, n+1 {
		fn(obj.Data[offset:offset+obj.W], n)
	}
}

func (obj *Volume) ScanRowsVolume(fn func(y, z int, f *Volume)) {
	offset := 0
	for z := 0; z < obj.D; z++ {
		for y := 0; y < obj.H; y++ {
			fn(y, z, obj.WrapRow(obj.Data[offset:offset+obj.W]))
			offset += obj.W
		}
	}
}

func (obj *Volume) ScanRows(fn func(y, z int, f []float64)) {
	offset := 0
	for z := 0; z < obj.D; z++ {
		for y := 0; y < obj.H; y++ {
			fn(y, z, obj.Data[offset:offset+obj.W])
			offset += obj.W
		}
	}
}

func (obj *Volume) Scan(fn func(x, y, z int, offset int, v float64)) {
	offset := 0
	for z := 0; z < obj.D; z++ {
		for y := 0; y < obj.H; y++ {
			for x := 0; x < obj.W; x++ {
				fn(x, y, z, offset, obj.Data[offset])
				offset++
			}
		}
	}
}

func (obj *Volume) ScanMatrices(fn func(z int, f []float64)) {
	offset := 0
	for z := 0; z < obj.D; z++ {
		fn(z, obj.Data[offset:offset+obj.W*obj.H])
		offset += obj.W * obj.H
	}
}

func (obj *Volume) GetRowByAbsoluteIndex(y int) *Volume {
	return WrapVolume(obj.W, 1, 1, obj.Data[y*obj.W:(y+1)*obj.W])
}

func (obj *Volume) GetRow(y, z int) *Volume {
	return WrapVolume(obj.W, 1, 1, obj.Data[z*obj.W*obj.H+y*obj.W:z*obj.W*obj.H+(y+1)*obj.W])
}

func (obj *Volume) GetRows(z int) *Volume {
	return WrapVolume(obj.W, obj.H, 1, obj.Data[z*obj.W*obj.H:(z+1)*obj.W*obj.H])
}

func (obj *Volume) String() string {
	var result string
	var offset int

	result += "\n"
	for z := 0; z < obj.D; z++ {
		//result += "\t[\n"
		for y := 0; y < obj.H; y++ {
			result += "\t\t["
			for x := 0; x < obj.W; x++ {
				if x > 0 {
					result += ", "
				}

				if obj.Data[offset] >= 0 {
					result += fmt.Sprintf(" %.9f", obj.Data[offset])
				} else {
					result += fmt.Sprintf("%.9f", obj.Data[offset])
				}
				offset++
			}
			result += " ]"
			if y < obj.H-1 {
				result += "\n"
			}
		}
		//result += "\t]\n"
	}
	return result
}
