package activation

import (
	"testing"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"

	"github.com/atkhx/nnet/data"
)

func BenchmarkTahn_Forward(b *testing.B) {
	layer := NewTahn()
	layer.InitDataSizes(28, 28, 28)

	input := &data.Data{}
	input.Init3DRandom(28, 28, 28, -1, 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Forward(input)
	}
}

func BenchmarkTahn_Backward(b *testing.B) {
	layer := NewTahn()

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

func TestTahn_Forward(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	layer := NewTahn()
	layer.InitDataSizes(2, 1, 1)

	inputs := data.NewVector(0.07, -0.3)
	output := layer.Forward(inputs)

	expected := &data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{0.06988589031642899, -0.2913126124515909},
	}

	assert.Equal(t, expected, output)
	assert.Equal(t, output, layer.GetOutput())
}

func TestTahn_Backward(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	layer := NewTahn()
	layer.InitDataSizes(2, 1, 1)
	layer.Forward(&data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{0.07, -0.3},
	})

	expected := &data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{-0.0895604366101212, 0.0008236232656439662},
	}

	assert.Equal(t, expected, layer.Backward(&data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{-0.09, 0.0009},
	}))

	assert.Equal(t, expected, layer.GetInputGradients())
}
