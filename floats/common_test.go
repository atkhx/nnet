package floats

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFill(t *testing.T) {
	type args struct {
		dst   []float64
		value float64
	}
	tests := []struct {
		name string
		args args
		want []float64
	}{
		{
			name: "empty",
			args: args{},
		},
		{
			name: "setValue",
			args: args{
				dst:   []float64{1, 2, 3, 4},
				value: 1,
			},
			want: []float64{1, 1, 1, 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			Fill(tt.args.dst, tt.args.value)
			assert.Equal(t, tt.want, tt.args.dst)
		})
	}
}

func TestFillRandom(t *testing.T) {
	type args struct {
		dst []float64
		min float64
		max float64
	}
	tests := []struct {
		name string
		args args
		seed int64
		want []float64
	}{
		{
			name: "empty",
		},
		{
			name: "zeroToOne",
			args: args{
				dst: []float64{1, 2, 3, 4, 5},
				min: 0,
				max: 1,
			},
			seed: 1,
			want: []float64{
				0.6046602879796196,
				0.9405090880450124,
				0.6645600532184904,
				0.4377141871869802,
				0.4246374970712657,
			},
		},
		{
			name: "minusOneToOne",
			args: args{
				dst: []float64{1, 2, 3, 4, 5},
				min: -1,
				max: 1,
			},
			seed: 1,
			want: []float64{
				0.20932057595923914,
				0.8810181760900249,
				0.32912010643698086,
				-0.12457162562603963,
				-0.15072500585746862,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rand.Seed(tt.seed)
			FillRandom(tt.args.dst, tt.args.min, tt.args.max)
			assert.Equal(t, tt.want, tt.args.dst)
		})
	}
}

func TestAddTo(t *testing.T) {
	type args struct {
		dst []float64
		src [][]float64
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
				dst: []float64{1, 2, 3, 4},
				src: [][]float64{
					{2, 3, 4, 5},
					{3, 4, 5, 6},
				},
			},
			want: []float64{
				1 + 2 + 3,
				2 + 3 + 4,
				3 + 4 + 5,
				4 + 5 + 6,
			},
		},
		{
			name: "emptySources",
			args: args{
				dst: []float64{1, 2, 3, 4},
			},
			want: []float64{1, 2, 3, 4},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			AddTo(tt.args.dst, tt.args.src...)
			assert.Equal(t, tt.want, tt.args.dst)
		})
	}
}

func TestAdd(t *testing.T) {
	type args struct {
		src [][]float64
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
			name: "oneSource",
			args: args{
				src: [][]float64{{1, 2, 3, 4}},
			},
			want: []float64{1, 2, 3, 4},
		},
		{
			name: "someSource",
			args: args{
				src: [][]float64{
					{1, 2, 3, 4},
					{2, 3, 4, 5},
					{3, 4, 5, 6},
				},
			},
			want: []float64{
				1 + 2 + 3,
				2 + 3 + 4,
				3 + 4 + 5,
				4 + 5 + 6,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, Add(tt.args.src...))
		})
	}
}

func TestGetMaxValue(t *testing.T) {
	type args struct {
		data []float64
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
			name: "maxLast",
			args: args{data: []float64{-1, 0, 1, 2, 3}},
			want: 3,
		},
		{
			name: "maxFirst",
			args: args{data: []float64{5, -1, 0, 1, 2}},
			want: 5,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, GetMaxValue(tt.args.data))
		})
	}
}

func TestGetMinValue(t *testing.T) {
	type args struct {
		data []float64
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
			name: "minFirst",
			args: args{data: []float64{-1, 0, 1, 2, 3}},
			want: -1,
		},
		{
			name: "minLast",
			args: args{data: []float64{-1, 0, 1, 2, -2}},
			want: -2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, GetMinValue(tt.args.data))
		})
	}
}

func TestGetMinMaxValues(t *testing.T) {
	type args struct {
		data []float64
	}
	tests := []struct {
		name    string
		args    args
		wantMin float64
		wantMax float64
	}{
		{
			name: "empty",
		},
		{
			name:    "sortedAsc",
			args:    args{data: []float64{-1, 0, 1, 2, 3, 4, 5}},
			wantMin: -1,
			wantMax: 5,
		},
		{
			name:    "sortedDesc",
			args:    args{data: []float64{5, 4, 3, 2, 1, 0, -1}},
			wantMin: -1,
			wantMax: 5,
		},
		{
			name:    "oneElement",
			args:    args{data: []float64{1}},
			wantMin: 1,
			wantMax: 1,
		},
		{
			name:    "twoElements",
			args:    args{data: []float64{1, 2}},
			wantMin: 1,
			wantMax: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotMin, gotMax := GetMinMaxValues(tt.args.data)
			assert.Equal(t, tt.wantMin, gotMin)
			assert.Equal(t, tt.wantMax, gotMax)
		})
	}
}

func TestGetMinMaxValuesInRange(t *testing.T) {
	type args struct {
		data []float64
		from int
		to   int
	}
	tests := []struct {
		name    string
		args    args
		wantMin float64
		wantMax float64
	}{
		{
			name: "empty",
		},
		{
			name: "fullRange",
			args: args{
				data: []float64{1, 2, 3},
				from: 0,
				to:   3,
			},
			wantMin: 1,
			wantMax: 3,
		},
		{
			name: "noTail",
			args: args{
				data: []float64{0, 1, 2, 3, 4},
				from: 0,
				to:   3,
			},
			wantMin: 0,
			wantMax: 2,
		},
		{
			name: "noHead",
			args: args{
				data: []float64{0, 1, 2, 3, 4},
				from: 1,
				to:   5,
			},
			wantMin: 1,
			wantMax: 4,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotMin, gotMax := GetMinMaxValuesInRange(tt.args.data, tt.args.from, tt.args.to)
			assert.Equal(t, tt.wantMin, gotMin)
			assert.Equal(t, tt.wantMax, gotMax)
		})
	}
}

func TestSumElements(t *testing.T) {
	type args struct {
		src []float64
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
			name: "oneElement",
			args: args{src: []float64{3}},
			want: 3,
		},
		{
			name: "someElement",
			args: args{src: []float64{-3, -4, -5}},
			want: -12,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, SumElements(tt.args.src))
		})
	}
}
