package activation

import (
	"testing"

	"github.com/atkhx/nnet/data"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

func BenchmarkRelu_Activate(b *testing.B) {
	layer := NewReLu()
	layer.InitDataSizes(28, 28, 28)

	input := &data.Data{}
	input.Init3DRandom(28, 28, 28, -1, 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Forward(input)
	}
}

func BenchmarkRelu_Backward(b *testing.B) {
	layer := NewReLu()

	ow, oh, od := layer.InitDataSizes(28, 28, 28)

	input := &data.Data{}
	input.Init3DRandom(28, 28, 28, -1, 1)

	deltas := &data.Data{}
	deltas.Init3DRandom(ow, oh, od, -1, 1)

	layer.Forward(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Backward(deltas)
	}
}

func TestRelu_Activate(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	layer := NewReLu()
	layer.InitDataSizes(2, 1, 1)

	inputs := data.NewVector(0.07, -0.3)
	output := layer.Forward(inputs)

	expected := &data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{0.07, 0.0},
	}

	assert.Equal(t, expected, output)
	assert.Equal(t, output, layer.GetOutput())
}

func TestRelu_Backward(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	layer := NewReLu()
	layer.InitDataSizes(2, 1, 1)
	layer.Forward(&data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{0.07, -0.3},
	})

	expected := &data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{-0.09, 0},
	}

	assert.Equal(t, expected, layer.Backward(&data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{-0.09, 0.0009},
	}))

	assert.Equal(t, expected, layer.GetInputGradients())
}
