package maxpooling

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/atkhx/nnet/data"
)

func TestMaxPooling_Forward(t *testing.T) {
	layer := New(FilterSize(2), Stride(2))
	layer.InitDataSizes(4, 4, 1)

	inputs := &data.Data{
		Dims: []int{4, 4, 0},
		Data: []float64{
			1.0, 3.0, 0.9, 0.1,
			0.5, 0.7, 1.0, 0.9,
			0.8, 0.1, 0.0, 0.7,
			0.3, 0.8, 0.3, 0.1,
		},
	}

	expected := &data.Data{
		Dims: []int{2, 2, 1},
		Data: []float64{
			3.0, 1.0,
			0.8, 0.7,
		},
	}

	assert.Equal(t, expected, layer.Forward(inputs))
}

func TestMaxPooling_Backward(t *testing.T) {
	layer := New(FilterSize(2), Stride(2))
	layer.InitDataSizes(4, 4, 1)

	assert.Equal(
		t,
		&data.Data{
			Dims: []int{2, 2, 1},
			Data: []float64{
				3.0, 0.9,
				0.8, 0.7,
			},
		},
		layer.Forward(&data.Data{
			Dims: []int{4, 4, 0},
			Data: []float64{
				1.0, 3.0, 0.9, 0.1,
				0.5, 0.7, 0.0, 0.9,
				0.8, 0.1, 0.0, 0.7,
				0.3, 0.8, 0.3, 0.1,
			},
		}),
	)

	assert.Equal(
		t,
		&data.Data{
			Dims: []int{4, 4, 1},
			Data: []float64{
				0.0, 1.0, 2.0, 0.0,
				0.0, 0.0, 0.0, 0.0,
				3.0, 0.0, 0.0, 4.0,
				0.0, 0.0, 0.0, 0.0,
			},
		},
		layer.Backward(&data.Data{
			Dims: []int{2, 2, 1},
			Data: []float64{
				1.0, 2.0,
				3.0, 4.0,
			},
		}),
	)
}
