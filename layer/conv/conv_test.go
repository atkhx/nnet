package conv

import (
	"github.com/atkhx/nnet/floats"
	"testing"

	"github.com/atkhx/nnet/data"
	"github.com/stretchr/testify/assert"
)

func BenchmarkConv_Forward(b *testing.B) {
	conv := New(FilterSize(5), FiltersCount(20))
	conv.InitDataSizes(34, 34, 20)

	input := &data.Data{}
	input.Init3DRandom(34, 34, 20, -1, 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		conv.Forward(input)
	}
}

func BenchmarkConv_Backward(b *testing.B) {
	conv := New(FilterSize(5), FiltersCount(20))
	ow, oh, od := conv.InitDataSizes(34, 34, 20)

	input := &data.Data{}
	input.Init3DRandom(34, 34, 20, -1, 1)

	deltas := &data.Data{}
	deltas.Init3DRandom(ow, oh, od, -1, 1)

	conv.Forward(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		conv.Backward(deltas)
	}
}

func TestConv_OneLayerFilter2x2WithoutPadding(t *testing.T) {
	conv := New(FilterSize(2), FiltersCount(1))
	conv.InitDataSizes(3, 3, 3)

	w11, w12, w13, w14 := 1.0, 2.0, 3.0, 4.0
	w21, w22, w23, w24 := 5.0, 6.0, 7.0, 8.0
	w31, w32, w33, w34 := 9.0, 10.0, 11.0, 12.0

	b1 := 0.1

	i11, i12, i13, i14, i15, i16, i17, i18, i19 := 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0
	i21, i22, i23, i24, i25, i26, i27, i28, i29 := 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0
	i31, i32, i33, i34, i35, i36, i37, i38, i39 := 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0

	conv.Weights.Init3DWithData(2, 2, 3, []float64{
		w11, w12,
		w13, w14,

		w21, w22,
		w23, w24,

		w31, w32,
		w33, w34,
	})
	conv.Biases.Data[0] = b1

	input := &data.Data{
		Dims: []int{3, 3, 3},
		Data: []float64{
			i11, i12, i13,
			i14, i15, i16,
			i17, i18, i19,

			i21, i22, i23,
			i24, i25, i26,
			i27, i28, i29,

			i31, i32, i33,
			i34, i35, i36,
			i37, i38, i39,
		},
	}

	y11 := b1 +
		(i11*w11 + i12*w12 + i14*w13 + i15*w14) +
		(i21*w21 + i22*w22 + i24*w23 + i25*w24) +
		(i31*w31 + i32*w32 + i34*w33 + i35*w34)

	y12 := b1 +
		(i12*w11 + i13*w12 + i15*w13 + i16*w14) +
		(i22*w21 + i23*w22 + i25*w23 + i26*w24) +
		(i32*w31 + i33*w32 + i35*w33 + i36*w34)

	y13 := b1 +
		(i14*w11 + i15*w12 + i17*w13 + i18*w14) +
		(i24*w21 + i25*w22 + i27*w23 + i28*w24) +
		(i34*w31 + i35*w32 + i37*w33 + i38*w34)

	y14 := b1 +
		(i15*w11 + i16*w12 + i18*w13 + i19*w14) +
		(i25*w21 + i26*w22 + i28*w23 + i29*w24) +
		(i35*w31 + i36*w32 + i38*w33 + i39*w34)

	out := conv.Forward(input)
	assert.Equal(t, []float64{
		y11, y12,
		y13, y14,
	}, out.Data)

	d11, d12, d13, d14 := 0.1, 0.2, 0.3, 0.4

	deltas := &data.Data{}
	deltas.InitMatrixWithData(2, 2, []float64{
		d11, d12,
		d13, d14,
	})

	igrad := conv.Backward(deltas)

	// weight rotated
	//wr := []float64{
	//	w14, w13,
	//	w12, w11,
	//}
	// deltas padded
	//dp := []float64{
	//	0.0, 0.0, 0.0, 0.0,
	//	0.0, d11, d12, 0.0,
	//	0.0, d13, d14, 0.0,
	//	0.0, 0.0, 0.0, 0.0,
	//}

	ig11 := w14*0.0 + w13*0.0 + w12*0.0 + w11*d11
	ig21 := w24*0.0 + w23*0.0 + w22*0.0 + w21*d11
	ig31 := w34*0.0 + w33*0.0 + w32*0.0 + w31*d11

	ig12 := w14*0.0 + w13*0.0 + w12*d11 + w11*d12
	ig22 := w24*0.0 + w23*0.0 + w22*d11 + w21*d12
	ig32 := w34*0.0 + w33*0.0 + w32*d11 + w31*d12

	ig13 := w14*0.0 + w13*0.0 + w12*d12 + w11*0.0
	ig23 := w24*0.0 + w23*0.0 + w22*d12 + w21*0.0
	ig33 := w34*0.0 + w33*0.0 + w32*d12 + w31*0.0

	ig14 := w14*0.0 + w13*d11 + w12*0.0 + w11*d13
	ig24 := w24*0.0 + w23*d11 + w22*0.0 + w21*d13
	ig34 := w34*0.0 + w33*d11 + w32*0.0 + w31*d13

	ig15 := w14*d11 + w13*d12 + w12*d13 + w11*d14
	ig25 := w24*d11 + w23*d12 + w22*d13 + w21*d14
	ig35 := w34*d11 + w33*d12 + w32*d13 + w31*d14

	ig16 := w14*d12 + w13*0.0 + w12*d14 + w11*0.0
	ig26 := w24*d12 + w23*0.0 + w22*d14 + w21*0.0
	ig36 := w34*d12 + w33*0.0 + w32*d14 + w31*0.0

	ig17 := w14*0.0 + w13*d13 + w12*0.0 + w11*0.0
	ig27 := w24*0.0 + w23*d13 + w22*0.0 + w21*0.0
	ig37 := w34*0.0 + w33*d13 + w32*0.0 + w31*0.0

	ig18 := w14*d13 + w13*d14 + w12*0.0 + w11*0.0
	ig28 := w24*d13 + w23*d14 + w22*0.0 + w21*0.0
	ig38 := w34*d13 + w33*d14 + w32*0.0 + w31*0.0

	ig19 := w14*d14 + w13*0.0 + w12*0.0 + w11*0.0
	ig29 := w24*d14 + w23*0.0 + w22*0.0 + w21*0.0
	ig39 := w34*d14 + w33*0.0 + w32*0.0 + w31*0.0

	assert.Equal(t, &data.Data{
		Dims: []int{3, 3, 3},
		Data: []float64{
			ig11, ig12, ig13,
			ig14, ig15, ig16,
			ig17, ig18, ig19,

			ig21, ig22, ig23,
			ig24, ig25, ig26,
			ig27, ig28, ig29,

			ig31, ig32, ig33,
			ig34, ig35, ig36,
			ig37, ig38, ig39,
		},
	}, igrad)

	assert.Equal(t, igrad, conv.iGrads)

	assert.Equal(t, &data.Data{
		Dims: []int{1, 1, 1},
		Data: []float64{d11 + d12 + d13 + d14},
	}, conv.bGrads)

	//inputs := []float64{
	//	i11, i12, i13,
	//	i14, i15, i16,
	//	i17, i18, i19,
	//
	//	i21, i22, i23,
	//	i24, i25, i26,
	//	i27, i28, i29,
	//
	//	i31, i32, i33,
	//	i34, i35, i36,
	//	i37, i38, i39,
	//}
	// deltas padded
	//dp := []float64{
	//	d11, d12,
	//	d13, d14,
	//}

	gw11 := i11*d11 + i12*d12 + i14*d13 + i15*d14
	gw21 := i21*d11 + i22*d12 + i24*d13 + i25*d14
	gw31 := i31*d11 + i32*d12 + i34*d13 + i35*d14

	gw12 := i12*d11 + i13*d12 + i15*d13 + i16*d14
	gw22 := i22*d11 + i23*d12 + i25*d13 + i26*d14
	gw32 := i32*d11 + i33*d12 + i35*d13 + i36*d14

	gw13 := i14*d11 + i15*d12 + i17*d13 + i18*d14
	gw23 := i24*d11 + i25*d12 + i27*d13 + i28*d14
	gw33 := i34*d11 + i35*d12 + i37*d13 + i38*d14

	gw14 := i15*d11 + i16*d12 + i18*d13 + i19*d14
	gw24 := i25*d11 + i26*d12 + i28*d13 + i29*d14
	gw34 := i35*d11 + i36*d12 + i38*d13 + i39*d14

	assert.Equal(t, &data.Data{
		Dims: []int{2, 2, 3},
		Data: []float64{
			gw11, gw12,
			gw13, gw14,

			gw21, gw22,
			gw23, gw24,

			gw31, gw32,
			gw33, gw34,
		},
	}, conv.wGrads)
}

func TestConv_OneLayerFilter2x2WithPadding1(t *testing.T) {
	conv := New(FilterSize(2), FiltersCount(1), Padding(1))
	conv.InitDataSizes(3, 3, 3)

	w11, w12, w13, w14 := 1.0, 2.0, 3.0, 4.0
	w21, w22, w23, w24 := 5.0, 6.0, 7.0, 8.0
	w31, w32, w33, w34 := 9.0, 10.0, 11.0, 12.0

	b1 := 0.1

	i11, i12, i13, i14, i15, i16, i17, i18, i19 := 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0
	i21, i22, i23, i24, i25, i26, i27, i28, i29 := 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0
	i31, i32, i33, i34, i35, i36, i37, i38, i39 := 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0

	conv.Weights.Init3DWithData(2, 2, 3, []float64{
		w11, w12,
		w13, w14,

		w21, w22,
		w23, w24,

		w31, w32,
		w33, w34,
	})
	conv.Biases.Data[0] = b1

	input := &data.Data{
		Dims: []int{3, 3, 3},
		Data: []float64{
			i11, i12, i13,
			i14, i15, i16,
			i17, i18, i19,

			i21, i22, i23,
			i24, i25, i26,
			i27, i28, i29,

			i31, i32, i33,
			i34, i35, i36,
			i37, i38, i39,
		},
	}

	//			0.0, 0.0, 0.0, 0.0, 0.0,
	//			0.0, i11, i12, i13, 0.0,

	y11 := b1 +
		(0.0*w11 + 0.0*w12 + 0.0*w13 + i11*w14) +
		(0.0*w21 + 0.0*w22 + 0.0*w23 + i21*w24) +
		(0.0*w31 + 0.0*w32 + 0.0*w33 + i31*w34)

	y12 := b1 +
		(0.0*w11 + 0.0*w12 + i11*w13 + i12*w14) +
		(0.0*w21 + 0.0*w22 + i21*w23 + i22*w24) +
		(0.0*w31 + 0.0*w32 + i31*w33 + i32*w34)

	y13 := b1 +
		(0.0*w11 + 0.0*w12 + i12*w13 + i13*w14) +
		(0.0*w21 + 0.0*w22 + i22*w23 + i23*w24) +
		(0.0*w31 + 0.0*w32 + i32*w33 + i33*w34)

	y14 := b1 +
		(0.0*w11 + 0.0*w12 + i13*w13 + 0.0*w14) +
		(0.0*w21 + 0.0*w22 + i23*w23 + 0.0*w24) +
		(0.0*w31 + 0.0*w32 + i33*w33 + 0.0*w34)

	// 			0.0, i11, i12, i13, 0.0,
	//			0.0, i14, i15, i16, 0.0,

	y15 := b1 +
		(0.0*w11 + i11*w12 + 0.0*w13 + i14*w14) +
		(0.0*w21 + i21*w22 + 0.0*w23 + i24*w24) +
		(0.0*w31 + i31*w32 + 0.0*w33 + i34*w34)

	y16 := b1 +
		(i11*w11 + i12*w12 + i14*w13 + i15*w14) +
		(i21*w21 + i22*w22 + i24*w23 + i25*w24) +
		(i31*w31 + i32*w32 + i34*w33 + i35*w34)

	y17 := b1 +
		(i12*w11 + i13*w12 + i15*w13 + i16*w14) +
		(i22*w21 + i23*w22 + i25*w23 + i26*w24) +
		(i32*w31 + i33*w32 + i35*w33 + i36*w34)

	y18 := b1 +
		(i13*w11 + 0.0*w12 + i16*w13 + 0.0*w14) +
		(i23*w21 + 0.0*w22 + i26*w23 + 0.0*w24) +
		(i33*w31 + 0.0*w32 + i36*w33 + 0.0*w34)

	//			0.0, i14, i15, i16, 0.0,
	//			0.0, i17, i18, i19, 0.0,

	y19 := b1 +
		(0.0*w11 + i14*w12 + 0.0*w13 + i17*w14) +
		(0.0*w21 + i24*w22 + 0.0*w23 + i27*w24) +
		(0.0*w31 + i34*w32 + 0.0*w33 + i37*w34)

	y20 := b1 +
		(i14*w11 + i15*w12 + i17*w13 + i18*w14) +
		(i24*w21 + i25*w22 + i27*w23 + i28*w24) +
		(i34*w31 + i35*w32 + i37*w33 + i38*w34)

	y21 := b1 +
		(i15*w11 + i16*w12 + i18*w13 + i19*w14) +
		(i25*w21 + i26*w22 + i28*w23 + i29*w24) +
		(i35*w31 + i36*w32 + i38*w33 + i39*w34)

	y22 := b1 +
		(i16*w11 + 0.0*w12 + i19*w13 + 0.0*w14) +
		(i26*w21 + 0.0*w22 + i29*w23 + 0.0*w24) +
		(i36*w31 + 0.0*w32 + i39*w33 + 0.0*w34)

	//			0.0, i17, i18, i19, 0.0,
	//			0.0, 0.0, 0.0, 0.0, 0.0

	y23 := b1 +
		(0.0*w11 + i17*w12 + 0.0*w13 + 0.0*w14) +
		(0.0*w21 + i27*w22 + 0.0*w23 + 0.0*w24) +
		(0.0*w31 + i37*w32 + 0.0*w33 + 0.0*w34)

	y24 := b1 +
		(i17*w11 + i18*w12 + 0.0*w13 + 0.0*w14) +
		(i27*w21 + i28*w22 + 0.0*w23 + 0.0*w24) +
		(i37*w31 + i38*w32 + 0.0*w33 + 0.0*w34)

	y25 := b1 +
		(i18*w11 + i19*w12 + 0.0*w13 + 0.0*w14) +
		(i28*w21 + i29*w22 + 0.0*w23 + 0.0*w24) +
		(i38*w31 + i39*w32 + 0.0*w33 + 0.0*w34)

	y26 := b1 +
		(i19*w11 + 0.0*w12 + 0.0*w13 + 0.0*w14) +
		(i29*w21 + 0.0*w22 + 0.0*w23 + 0.0*w24) +
		(i39*w31 + 0.0*w32 + 0.0*w33 + 0.0*w34)

	out := conv.Forward(input)
	assert.Equal(t, []float64{
		y11, y12, y13, y14,
		y15, y16, y17, y18,
		y19, y20, y21, y22,
		y23, y24, y25, y26,
	}, out.Data)

	//d11, d12, d13, d14 := 0.1, 0.2, 0.3, 0.4
	//d15, d16, d17, d18 := 0.1, 0.2, 0.3, 0.4

	var d11, d12, d13, d14,
		d15, d16, d17, d18,
		d19, d20, d21, d22,
		d23, d24, d25, d26 = 0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 1.0, 1.1, 1.2,
		1.3, 1.4, 1.5, 1.6

	deltas := &data.Data{}
	deltas.InitMatrixWithData(4, 4, []float64{
		d11, d12, d13, d14,
		d15, d16, d17, d18,
		d19, d20, d21, d22,
		d23, d24, d25, d26,
	})

	igrad := conv.Backward(deltas)

	// weight rotated
	//wr := []float64{
	//	w14, w13,
	//	w12, w11,
	//}
	// deltas padded
	//dp := []float64{
	//	d11, d12, d13, d14,
	//	d15, d16, d17, d18,
	//	d19, d20, d21, d22,
	//	d23, d24, d25, d26,
	//}

	ig11 := d11*w14 + d12*w13 + d15*w12 + d16*w11
	ig21 := d16*w21 + d15*w22 + d12*w23 + d11*w24

	//ig31 := d11*w34 + d12*w33 + d15*w32 + d16*w31
	ig31 := d16*w31 + d15*w32 + d12*w33 + d11*w34

	ig12 := d12*w14 + d13*w13 + d16*w12 + d17*w11
	//ig22 := d12*w24 + d13*w23 + d16*w22 + d17*w21
	ig22 := d17*w21 + d16*w22 + d13*w23 + +d12*w24
	ig32 := d12*w34 + d13*w33 + d16*w32 + d17*w31

	ig13 := d13*w14 + d14*w13 + d17*w12 + d18*w11
	//ig23 := d13*w24 + d14*w23 + d17*w22 + d18*w21
	ig23 := d18*w21 + d17*w22 + d14*w23 + d13*w24
	//ig33 := d13*w34 + d14*w33 + d17*w32 + d18*w31
	ig33 := d18*w31 + d17*w32 + d14*w33 + d13*w34

	// 2nd line

	ig14 := d15*w14 + d16*w13 + d19*w12 + d20*w11
	ig24 := d15*w24 + d16*w23 + d19*w22 + d20*w21
	ig34 := d15*w34 + d16*w33 + d19*w32 + d20*w31

	ig15 := d16*w14 + d17*w13 + d20*w12 + d21*w11
	ig25 := d16*w24 + d17*w23 + d20*w22 + d21*w21
	ig35 := d16*w34 + d17*w33 + d20*w32 + d21*w31

	// ig16 - invalid
	//ig16 := d17*w14 + d18*w13 + d21*w12 + d22*w11
	ig16 := d22*w11 + d21*w12 + d18*w13 + d17*w14
	//ig26 := d17*w24 + d18*w23 + d21*w22 + d22*w21
	ig26 := d22*w21 + d21*w22 + d18*w23 + +d17*w24
	ig36 := d17*w34 + d18*w33 + d21*w32 + d22*w31

	// 3rd line

	ig17 := d19*w14 + d20*w13 + d23*w12 + d24*w11
	ig27 := d19*w24 + d20*w23 + d23*w22 + d24*w21
	//ig37 := d19*w34 + d20*w33 + d23*w32 + d24*w31
	ig37 := d24*w31 + d23*w32 + d20*w33 + d19*w34

	//ig18 := d20*w14 + d21*w13 + d24*w12 + d25*w11
	// ig16 := d22*w11 + d21*w12 + d18*w13 + d17*w14
	ig18 := d25*w11 + d24*w12 + d21*w13 + d20*w14
	ig28 := d20*w24 + d21*w23 + d24*w22 + d25*w21
	ig38 := d20*w34 + d21*w33 + d24*w32 + d25*w31

	ig19 := d21*w14 + d22*w13 + d25*w12 + d26*w11
	ig29 := d21*w24 + d22*w23 + d25*w22 + d26*w21
	ig39 := d21*w34 + d22*w33 + d25*w32 + d26*w31

	assert.Equal(t, &data.Data{
		Dims: []int{3, 3, 3},
		Data: []float64{ // 6
			ig11, ig12, ig13,
			ig14, ig15, ig16,
			ig17, ig18, ig19,

			ig21, ig22, ig23,
			ig24, ig25, ig26,
			ig27, ig28, ig29,

			ig31, ig32, ig33,
			ig34, ig35, ig36,
			ig37, ig38, ig39,
		},
	}, igrad)

	assert.Equal(t, &data.Data{
		Dims: []int{1, 1, 1},
		Data: []float64{
			d11 + d12 + d13 + d14 +
				d15 + d16 + d17 + d18 +
				d19 + d20 + d21 + d22 +
				d23 + d24 + d25 + d26,
		},
	}, conv.bGrads)

	//inputs := []float64{
	//	0.0, 0.0, 0.0, 0.0, 0.0,
	//	0.0, i11, i12, i13, 0.0,
	//	0.0, i14, i15, i16, 0.0,
	//	0.0, i17, i18, i19, 0.0,
	//	0.0, 0.0, 0.0, 0.0, 0.0,
	//
	//	0.0, 0.0, 0.0, 0.0, 0.0,
	//	0.0, i21, i22, i23, 0.0,
	//	0.0, i24, i25, i26, 0.0,
	//	0.0, i27, i28, i29, 0.0,
	//	0.0, 0.0, 0.0, 0.0, 0.0,
	//
	//	0.0, 0.0, 0.0, 0.0, 0.0,
	//	0.0, i31, i32, i33, 0.0,
	//	0.0, i34, i35, i36, 0.0,
	//	0.0, i37, i38, i39, 0.0,
	//	0.0, 0.0, 0.0, 0.0, 0.0,
	//}
	// deltas padded
	//dp := []float64{
	//	d11, d12, d13, d14,
	//	d15, d16, d17, d18,
	//	d19, d20, d21, d22,
	//	d23, d24, d25, d26,
	//}

	gw11 := 0.0 +
		0.0*d11 + 0.0*d12 + 0.0*d13 + 0.0*d14 +
		0.0*d15 + i11*d16 + i12*d17 + i13*d18 +
		0.0*d19 + i14*d20 + i15*d21 + i16*d22 +
		0.0*d23 + i17*d24 + i18*d25 + i19*d26

	gw21 := 0.0 +
		0.0*d11 + 0.0*d12 + 0.0*d13 + 0.0*d14 +
		0.0*d15 + i21*d16 + i22*d17 + i23*d18 +
		0.0*d19 + i24*d20 + i25*d21 + i26*d22 +
		0.0*d23 + i27*d24 + i28*d25 + i29*d26

	gw31 := 0.0 +
		0.0*d11 + 0.0*d12 + 0.0*d13 + 0.0*d14 +
		0.0*d15 + i31*d16 + i32*d17 + i33*d18 +
		0.0*d19 + i34*d20 + i35*d21 + i36*d22 +
		0.0*d23 + i37*d24 + i38*d25 + i39*d26

	gw12 := 0.0 +
		0.0*d11 + 0.0*d12 + 0.0*d13 + 0.0*d14 +
		i11*d15 + i12*d16 + i13*d17 + 0.0*d18 +
		i14*d19 + i15*d20 + i16*d21 + 0.0*d22 +
		i17*d23 + i18*d24 + i19*d25 + 0.0*d26

	gw22 := 0.0 +
		0.0*d11 + 0.0*d12 + 0.0*d13 + 0.0*d14 +
		i21*d15 + i22*d16 + i23*d17 + 0.0*d18 +
		i24*d19 + i25*d20 + i26*d21 + 0.0*d22 +
		i27*d23 + i28*d24 + i29*d25 + 0.0*d26

	gw32 := 0.0 +
		0.0*d11 + 0.0*d12 + 0.0*d13 + 0.0*d14 +
		i31*d15 + i32*d16 + i33*d17 + 0.0*d18 +
		i34*d19 + i35*d20 + i36*d21 + 0.0*d22 +
		i37*d23 + i38*d24 + i39*d25 + 0.0*d26

	gw13 := 0.0 +
		0.0*d11 + i11*d12 + i12*d13 + i13*d14 +
		0.0*d15 + i14*d16 + i15*d17 + i16*d18 +
		0.0*d19 + i17*d20 + i18*d21 + i19*d22 +
		0.0*d23 + 0.0*d24 + 0.0*d25 + 0.0*d26

	gw23 := 0.0 +
		0.0*d11 + i21*d12 + i22*d13 + i23*d14 +
		0.0*d15 + i24*d16 + i25*d17 + i26*d18 +
		0.0*d19 + i27*d20 + i28*d21 + i29*d22 +
		0.0*d23 + 0.0*d24 + 0.0*d25 + 0.0*d26

	gw33 := 0.0 +
		0.0*d11 + i31*d12 + i32*d13 + i33*d14 +
		0.0*d15 + i34*d16 + i35*d17 + i36*d18 +
		0.0*d19 + i37*d20 + i38*d21 + i39*d22 +
		0.0*d23 + 0.0*d24 + 0.0*d25 + 0.0*d26

	gw14 := 0.0 +
		i11*d11 + i12*d12 + i13*d13 + 0.0*d14 +
		i14*d15 + i15*d16 + i16*d17 + 0.0*d18 +
		i17*d19 + i18*d20 + i19*d21 + 0.0*d22 +
		0.0*d23 + 0.0*d24 + 0.0*d25 + 0.0*d26

	gw24 := 0.0 +
		i21*d11 + i22*d12 + i23*d13 + 0.0*d14 +
		i24*d15 + i25*d16 + i26*d17 + 0.0*d18 +
		i27*d19 + i28*d20 + i29*d21 + 0.0*d22 +
		0.0*d23 + 0.0*d24 + 0.0*d25 + 0.0*d26

	gw34 := 0.0 +
		i31*d11 + i32*d12 + i33*d13 + 0.0*d14 +
		i34*d15 + i35*d16 + i36*d17 + 0.0*d18 +
		i37*d19 + i38*d20 + i39*d21 + 0.0*d22 +
		0.0*d23 + 0.0*d24 + 0.0*d25 + 0.0*d26

	expected := &data.Data{
		Dims: []int{2, 2, 3},
		Data: []float64{
			gw11, gw12,
			gw13, gw14,

			gw21, gw22,
			gw23, gw24,

			gw31, gw32,
			gw33, gw34,
		},
	}

	floats.Round(expected.Data, 100000)
	floats.Round(conv.wGrads.Data, 100000)

	assert.Equal(t, expected, conv.wGrads)
}
