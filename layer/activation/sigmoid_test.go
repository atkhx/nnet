package activation

import (
	"testing"

	"github.com/atkhx/nnet/data"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

func BenchmarkSigmoid_Forward(b *testing.B) {
	layer := NewSigmoid()
	layer.InitDataSizes(28, 28, 28)

	input := &data.Data{}
	input.Init3DRandom(28, 28, 28, -1, 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Forward(input)
	}
}

func BenchmarkSigmoid_Backward(b *testing.B) {
	layer := NewSigmoid()

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

func TestSigmoid_Forward(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	layer := NewSigmoid()
	layer.InitDataSizes(2, 1, 1)

	inputs := data.NewVector(0.7, -0.7)
	output := layer.Forward(inputs)

	expected := &data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{0.6681877721681662, 0.3318122278318339},
	}

	assert.Equal(t, expected, output)
	assert.Equal(t, output, layer.GetOutput())
}

func TestSigmoid_Backward(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	layer := NewSigmoid()
	layer.InitDataSizes(2, 1, 1)
	layer.Forward(&data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{0.7, -0.7},
	})

	expected := &data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{-0.0022171287329310905, 0.002217128732931091},
	}

	assert.Equal(t, expected, layer.Backward(&data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{-0.01, 0.01},
	}))

	assert.Equal(t, expected, layer.GetInputGradients())
}
