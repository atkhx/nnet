package fc

import (
	"github.com/atkhx/nnet/floats"
	"testing"

	"github.com/atkhx/nnet/data"
	"github.com/stretchr/testify/assert"
)

func TestFC_InitDataSizes(t *testing.T) {
	iw, ih, id := 3, 3, 3
	ow, oh, od := 13, 14, 15

	layer := New(OutputSizes(ow, oh, od))

	OW, OH, OD := layer.InitDataSizes(iw, ih, id)

	assert.EqualValues(t, []int{ow, oh, od}, []int{OW, OH, OD})

	inputNeuronsCount := iw * ih * id
	outputNeuronsCount := ow * oh * od

	biases := layer.GetBiases()
	weights := layer.GetWeights()
	output := layer.GetOutput()

	b, gradBiases := layer.GetBiasesWithGradient()
	assert.Equal(t, b, biases)

	w, gradWeights := layer.GetWeightsWithGradient()
	assert.Equal(t, w, weights)

	assert.Equal(t, outputNeuronsCount, len(output.Data))

	assert.Equal(t, outputNeuronsCount*inputNeuronsCount, len(weights.Data))
	assert.Equal(t, outputNeuronsCount*inputNeuronsCount, len(gradWeights.Data))

	assert.Equal(t, outputNeuronsCount, len(biases.Data))
	assert.Equal(t, outputNeuronsCount, len(gradBiases.Data))
}

func TestFC(t *testing.T) {
	layer := New()
	layer.InitDataSizes(3, 3, 3)

	bias := 0.017
	layer.Biases.Data = []float64{bias}
	layer.Weights.Data = []float64{
		0.01, 0.02, 0.03,
		0.04, 0.05, 0.06,
		0.07, 0.08, 0.09,

		0.11, 0.12, 0.13,
		0.14, 0.15, 0.16,
		0.17, 0.18, 0.19,

		0.21, 0.22, 0.23,
		0.24, 0.25, 0.26,
		0.27, 0.28, 0.29,
	}

	inputs := &data.Data{}
	inputs.Init3D(3, 3, 3)
	inputs.Data = []float64{
		0.32, 0.33, 0.34,
		0.35, 0.36, 0.37,
		0.38, 0.39, 0.40,

		0.42, 0.43, 0.44,
		0.45, 0.46, 0.47,
		0.48, 0.49, 0.50,

		0.52, 0.53, 0.54,
		0.55, 0.56, 0.57,
		0.58, 0.59, 0.60,
	}

	expected := &data.Data{
		Dims: []int{1, 1, 1},
		Data: []float64{
			bias +
				//
				0.01*0.32 + 0.02*0.33 + 0.03*0.34 +
				0.04*0.35 + 0.05*0.36 + 0.06*0.37 +
				0.07*0.38 + 0.08*0.39 + 0.09*0.40 +
				//
				0.11*0.42 + 0.12*0.43 + 0.13*0.44 +
				0.14*0.45 + 0.15*0.46 + 0.16*0.47 +
				0.17*0.48 + 0.18*0.49 + 0.19*0.50 +
				//
				0.21*0.52 + 0.22*0.53 + 0.23*0.54 +
				0.24*0.55 + 0.25*0.56 + 0.26*0.57 +
				0.27*0.58 + 0.28*0.59 + 0.29*0.60,
		},
	}

	actualOut := layer.Forward(inputs)
	floats.Round(expected.Data, 100000)
	floats.Round(actualOut.Data, 100000)

	assert.Equal(t, expected, actualOut)
	//assert.Equal(t, expected, layer.GetOutput())

	deltas := &data.Data{}
	deltas.Init3D(1, 1, 1)
	deltas.Data = []float64{0.01}

	expectedGradients := &data.Data{
		Dims: []int{3, 3, 3},
		Data: []float64{
			0.0001, 0.0002, 0.0003,
			0.0004, 0.0005, 0.0006,
			0.0007, 0.0008, 0.0009,

			0.0011, 0.0012, 0.0013,
			0.0014, 0.0015, 0.0016,

			0.0017, 0.0018, 0.0019,
			0.0021, 0.0022, 0.0023,
			0.0024, 0.0025, 0.00260,
			0.0027, 0.0028, 0.0029,
		},
	}

	actualGradient := layer.Backward(deltas)
	floats.Round(expectedGradients.Data, 100000)
	floats.Round(actualGradient.Data, 100000)

	assert.Equal(t, expectedGradients, actualGradient)
	//assert.Equal(t, expectedGradients, layer.GetInputGradients())
}

func BenchmarkFC_Activate(b *testing.B) {
	fc := New()
	fc.InitDataSizes(100, 100, 100)

	input := &data.Data{}
	input.Init3DRandom(100, 100, 100, -1, 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fc.Forward(input)
	}
}

func BenchmarkFC_Backward(b *testing.B) {
	fc := New()

	ow, oh, od := fc.InitDataSizes(100, 100, 100)

	input := &data.Data{}
	input.Init3DRandom(100, 100, 100, -1, 1)

	deltas := &data.Data{}
	deltas.Init3DRandom(ow, oh, od, -1, 1)

	fc.Forward(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fc.Backward(deltas)
	}
}
