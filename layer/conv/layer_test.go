package conv

import (
	"github.com/atkhx/nnet/data"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestNew(t *testing.T) {
	layer := New()
	assert.Equal(t, defaultFilterWidth, layer.FWidth)
	assert.Equal(t, defaultFilterHeight, layer.FHeight)
	assert.Equal(t, defaultFiltersCount, layer.FCount)
}

func TestLayer(t *testing.T) {
	layer := New(FilterSize(2), FiltersCount(2), Stride(1))
	layer.InitDataSizes(3, 3, 2)

	inputs := &data.Data{}
	inputs.InitCube(3, 3, 2)
	inputs.Data = []float64{
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9,

		1.1, 1.2, 1.3,
		1.4, 1.5, 1.6,
		1.7, 1.8, 1.9,
	}

	layer.Weights.Data = []float64{
		// filter 1, chan 1 weights
		2.1, 2.2,
		2.4, 2.5,

		// filter 1, chan 2 weights
		3.1, 3.2,
		3.4, 3.5,

		// filter 2, chan 1 weights
		4.1, 4.2,
		4.3, 4.4,

		// filter 2, chan 2 weights
		5.1, 5.2,
		5.3, 5.4,
	}

	expectedOutput := &data.Data{
		Dims: []int{2, 2, 2},
		Data: []float64{
			24.22, 26.46,
			30.94, 33.18000000000001,

			36.739999999999995, 40.54,
			48.14, 51.94,
		},
	}

	layer.Biases.Data = []float64{4.1, 4.2}

	layer.Activate(inputs)

	// test correct activation algorithm
	assert.Equal(t, expectedOutput, layer.Activate(inputs))

	// check than output saved in layer
	assert.Equal(t, expectedOutput, layer.GetOutput())

	{ // check than input is link on original data
		inputs.Data[0] = 17.0

		assert.Equal(t, inputs, layer.GetInputs())
		// return changed input value
		inputs.Data[0] = 0.1
	}

	{ // check than GetWeights return mutable data
		weights := layer.GetWeights()
		weights.Data[0] = 18.0

		assert.Equal(t, weights, layer.GetWeights())
		// return changed weight
		weights.Data[0] = 2.1
	}

	// test backprop
	deltas := &data.Data{}
	deltas.InitCube(3, 3, 2)
	deltas.Data = []float64{
		0.0, 0.1, 0.0,
		0.1, 0.0, 0.0,
		0.0, 0.0, 0.9,

		1.0, 1.1, 0.0,
		1.1, 1.0, 0.0,
		0.0, 0.0, 1.0,
	}

	expectedGrads := &data.Data{}
	expectedGrads.InitCube(3, 3, 2)
	expectedGrads.Data = []float64{
		0, 0.21000000000000002, 0.22000000000000003, 0,
		0.45, 0.47000000000000003, 0,
		0.24, 0.25, 0,

		0.31000000000000005, 0.32000000000000006, 0,
		0.6500000000000001, 0.6700000000000002, 0,
		0.34, 0.35000000000000003,
	}

	// test backprop algorithm
	assert.Equal(t, expectedGrads, layer.Backprop(deltas))

	// check than input gradients saved in layer
	assert.Equal(t, expectedGrads, layer.GetInputGradients())

	// test than gradients not leak between backprop calls
	assert.Equal(t, expectedGrads, layer.Backprop(deltas))

	{

		expected := &data.Data{}
		expected.InitCube(2, 2, 4)
		expected.Data = []float64{
			0.07, 0.09,
			0.13, 0.15000000000000002,

			0.27, 0.29000000000000004,
			0.33000000000000007, 0.35000000000000003,

			0, 0,
			0, 0,

			0, 0,
			0, 0,
		}

		_, g := layer.GetWeightsWithGradient()
		assert.Equal(t, expected, g)
	}

	{
		expected := &data.Data{}
		expected.InitCube(2, 1, 1)
		expected.Data = []float64{0.2, 0}

		_, g := layer.GetBiasesWithGradient()
		assert.Equal(t, expected, g)
	}
}
