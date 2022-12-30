package floats

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestDot(t *testing.T) {
	type args struct {
		sliceA []float64
		sliceB []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "empty",
		},
		{
			name: "length less 4",
			args: args{
				sliceA: []float64{1, 2, 3},
				sliceB: []float64{5, 6, 7},
			},
			want: 5 + 12 + 21,
		},
		{
			name: "length equals 4",
			args: args{
				sliceA: []float64{1, 2, 3, 4},
				sliceB: []float64{5, 6, 7, 8},
			},
			want: 5 + 12 + 21 + 32,
		},
		{
			name: "length greater 4",
			args: args{
				sliceA: []float64{1, 2, 3, 4, 5},
				sliceB: []float64{5, 6, 7, 8, 9},
			},
			want: 5 + 12 + 21 + 32 + 45,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, Dot(tt.args.sliceA, tt.args.sliceB))
		})
	}
}

func TestMultiplyAndAddTo(t *testing.T) {
	type args struct {
		dst []float64
		src []float64
		k   float64
	}

	tests := []struct {
		name string
		args args
		want []float64
	}{
		{
			name: "empty",
		},
		{
			name: "sameSize",
			args: args{
				dst: []float64{1, 2, 3, 4, 5},
				src: []float64{2, 3, 4, 5, 6},
				k:   13,
			},
			want: []float64{
				1 + 2*13,
				2 + 3*13,
				3 + 4*13,
				4 + 5*13,
				5 + 6*13,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			MultiplyAndAddTo(tt.args.dst, tt.args.src, tt.args.k)
			assert.Equal(t, tt.want, tt.args.dst)
		})
	}
}
func TestMultiplyAndAdd(t *testing.T) {
	type args struct {
		src []float64
		k   float64
	}
	tests := []struct {
		name string
		args args
		want []float64
	}{
		{
			name: "empty",
			want: []float64{},
		},
		{
			name: "sameSize",
			args: args{
				src: []float64{2, 3, 4, 5, 6},
				k:   17,
			},
			want: []float64{2 * 17, 3 * 17, 4 * 17, 5 * 17, 6 * 17},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, MultiplyAndAdd(tt.args.src, tt.args.k))
		})
	}
}
