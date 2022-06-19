package activation

import (
	"testing"

	"github.com/atkhx/nnet/data"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

func TestLayer_Activate(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	activationFunc := NewMockActivation(ctrl)
	activationFunc.EXPECT().Forward(1.0).Return(0.40)
	activationFunc.EXPECT().Forward(0.3).Return(0.01)

	layer := New(activationFunc)
	layer.InitDataSizes(2, 1, 1)

	inputs := data.NewVector(1.0, 0.3)
	output := layer.Activate(inputs)

	expected := &data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{0.40, 0.01},
	}

	assert.Equal(t, expected, output)
	assert.Equal(t, output, layer.GetOutput())
}

func TestLayer_Backprop(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	activationFunc := NewMockActivation(ctrl)

	activationFunc.EXPECT().Forward(1.0).Return(0.40)
	activationFunc.EXPECT().Forward(0.3).Return(0.01)

	activationFunc.EXPECT().Backward(0.40).Return(0.7)
	activationFunc.EXPECT().Backward(0.01).Return(0.3)

	layer := New(activationFunc)
	layer.InitDataSizes(2, 1, 1)
	layer.Activate(&data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{1.0, 0.3},
	})

	expected := &data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{-0.063, 0.00027},
	}

	assert.Equal(t, expected, layer.Backprop(&data.Data{
		Dims: []int{2, 1, 1},
		Data: []float64{-0.09, 0.0009},
	}))

	assert.Equal(t, expected, layer.GetInputGradients())
}
