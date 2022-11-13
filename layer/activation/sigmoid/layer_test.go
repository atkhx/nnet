package sigmoid

import (
	"testing"

	"github.com/atkhx/nnet/data"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

func BenchmarkLayer_Activate(b *testing.B) {
	layer := New()
	layer.InitDataSizes(28, 28, 28)

	input := &data.Data{}
	input.InitCubeRandom(28, 28, 28, -1, 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Activate(input)
	}
}

func BenchmarkLayer_Backprop(b *testing.B) {
	layer := New()

	ow, oh, od := layer.InitDataSizes(28, 28, 28)

	input := &data.Data{}
	input.InitCubeRandom(28, 28, 28, -1, 1)

	deltas := &data.Data{}
	deltas.InitCubeRandom(ow, oh, od, -1, 1)

	layer.Activate(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Backprop(deltas)
	}
}

func TestLayer_Activate(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	layer := New()
	layer.InitDataSizes(2, 1, 1)

	inputs := data.NewVector(0.7, -0.7)
	output := layer.Activate(inputs)

	expected := &data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{0.6681877721681662, 0.3318122278318339},
	}

	assert.Equal(t, expected, output)
	assert.Equal(t, output, layer.GetOutput())
}

func TestLayer_Backprop(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	layer := New()
	layer.InitDataSizes(2, 1, 1)
	layer.Activate(&data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{0.7, -0.7},
	})

	expected := &data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{-0.0022171287329310905, 0.002217128732931091},
	}

	assert.Equal(t, expected, layer.Backprop(&data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{-0.01, 0.01},
	}))

	assert.Equal(t, expected, layer.GetInputGradients())
}
