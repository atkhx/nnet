package data

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestData_Mean(t *testing.T) {
	data := WrapData(3, 1, 1, []float64{0.7081, 0.3542, 0.1054})
	mean := data.Mean()

	assert.Equal(t, 0.38923, Round(mean.Data.At(0, 0, 0), 100_000))

	mean.Backward()
	assert.Equal(t, []float64{0.33333, 0.33333, 0.33333}, RoundFloats(data.Grad.Data, 100_000))
}

func TestData_ColMean(t *testing.T) {
	data := WrapData(3, 2, 1, []float64{
		0.7081, 0.3542, 0.1054,
		0.5996, 0.0904, 0.0899,
	})
	mean := data.ColMean()

	assert.Equal(t, []float64{0.6539, 0.2223, 0.0976}, RoundFloats(mean.Data.Data, 10_000))

	mean.Sum().Backward()

	assert.Equal(t, []float64{
		0.5, 0.5, 0.5,
		0.5, 0.5, 0.5,
	}, RoundFloats(data.Grad.Data, 10_000))
}

func TestData_Std(t *testing.T) {
	data := WrapData(3, 1, 1, []float64{0.7081, 0.3542, 0.1054})
	stdd := data.Std()

	assert.Equal(t, 0.30287, Round(stdd.Data.Data[0], 100_000))

	stdd.Backward()
	assert.Equal(t, []float64{0.52640, -0.05783, -0.46857}, RoundFloats(data.Grad.Data, 100_000))
}

func TestData_ColStd_3x2(t *testing.T) {
	data := WrapData(3, 2, 1, []float64{
		0.7081, 0.3542, 0.1054,
		0.5996, 0.0904, 0.0899,
	})
	std := data.ColStd()
	std.Sum().Backward()

	assert.Equal(t, []float64{
		0.0767, 0.1865, 0.0110,
	}, RoundFloats(std.Data.Data, 10_000))

	assert.Equal(t, []float64{
		0.7071, 0.7071, 0.7071,
		-0.7071, -0.7071, -0.7071,
	}, RoundFloats(data.Grad.Data, 10_000))
}

func TestData_ColStd_3x3(t *testing.T) {
	data := WrapData(3, 3, 1, []float64{
		0.7081, 0.3542, 0.1054,
		0.5996, 0.0904, 0.0899,
		0.8822, 0.9887, 0.0080,
	})
	std := data.ColStd()
	std.Sum().Backward()

	assert.Equal(t, []float64{
		0.1426, 0.4617, 0.0523,
	}, RoundFloats(std.Data.Data, 10_000))

	assert.Equal(t, []float64{
		-0.0767, -0.1338, 0.3595,
		-0.4572, -0.4195, 0.2115,
		0.5339, 0.5533, -0.5710,
	}, RoundFloats(data.Grad.Data, 10_000))
}

func TestData_Softmax(t *testing.T) {
	inputs := WrapData(3, 1, 1, []float64{
		0.7081, 0.3542, 0.1054,
	})

	target := WrapData(3, 1, 1, []float64{
		0.0, 1.0, 0.0,
	})

	softmax := inputs.Softmax()

	loss := softmax.Classification(target)
	loss.Backward()

	assert.Equal(t, 1.1645, Round(loss.Data.Data[0], 10_000))
	assert.Equal(t, 1.0, Round(softmax.Sum().Data.Data[0], 10))

	assert.Equal(t, []float64{
		0.4446, 0.3121, 0.2433,
	}, RoundFloats(softmax.Data.Data, 10_000))

	assert.Equal(t, []float64{
		0.4446, -0.6879, 0.2433,
	}, RoundFloats(inputs.Grad.Data, 10_000))
}

func TestData_Classification(t *testing.T) {
	inputs := WrapData(3, 1, 1, []float64{
		0.7081, 0.3542, 0.1054,
	})

	target := WrapData(3, 1, 1, []float64{
		1.0, 0.0, 0.0,
	})

	softmax := inputs.Softmax()

	loss := softmax.Classification(target)
	loss.Backward()

	assert.Equal(t, []float64{
		0.8106,
	}, RoundFloats(loss.Data.Data, 10_000))

	assert.Equal(t, []float64{
		0.4446, 0.3121, 0.2433,
	}, RoundFloats(softmax.Data.Data, 10_000))

	assert.Equal(t, []float64{
		-2.2493, 0, 0,
	}, RoundFloats(softmax.Grad.Data, 10_000))

	assert.Equal(t, []float64{
		-0.5554, 0.3121, 0.2433,
	}, RoundFloats(inputs.Grad.Data, 10_000))
}

func TestData_CrossEntropy(t *testing.T) {
	inputs := WrapData(3, 1, 1, []float64{
		0.7081, 0.3542, 0.1054,
	})

	target := WrapData(3, 1, 1, []float64{
		1.0, 0.0, 0.0,
	})

	loss := inputs.CrossEntropy(target)
	loss.Backward()

	assert.Equal(t, []float64{
		0.8106,
	}, RoundFloats(loss.Data.Data, 10_000))

	assert.Equal(t, []float64{
		-0.5554, 0.3121, 0.2433,
	}, RoundFloats(inputs.Grad.Data, 10_000))
}

func TestData_MulRowVector(t *testing.T) {
	input := WrapData(3, 3, 1, []float64{
		0.7081, 0.3542, 0.1054,
		0.5996, 0.0904, 0.0899,
		0.8822, 0.9887, 0.0080,
	})

	weights := WrapData(3, 1, 1, []float64{
		0.2908, 0.7408, 0.4012,
	})

	result := input.MulRowVector(weights)

	sum := result.Sum()
	sum.Backward()

	assert.Equal(t, []float64{
		0.2059, 0.2624, 0.0423,
		0.1744, 0.0670, 0.0361,
		0.2565, 0.7324, 0.0032,
	}, RoundFloats(result.Data.Data, 10_000))

	assert.Equal(t, 1.7802, Round(sum.Data.Data[0], 10_000))

	assert.Equal(t, []float64{
		0.2908, 0.7408, 0.4012,
		0.2908, 0.7408, 0.4012,
		0.2908, 0.7408, 0.4012,
	}, RoundFloats(input.Grad.Data, 10_000))

	assert.Equal(t, []float64{
		2.1899, 1.4333, 0.2033,
	}, RoundFloats(weights.Grad.Data, 10_000))
}

func TestData_DivRowVector(t *testing.T) {
	input := WrapData(3, 3, 1, []float64{
		0.7081, 0.3542, 0.1054,
		0.5996, 0.0904, 0.0899,
		0.8822, 0.9887, 0.0080,
	})

	weights := WrapData(3, 1, 1, []float64{
		0.2908, 0.7408, 0.4012,
	})

	result := input.DivRowVector(weights)

	sum := result.Sum()
	sum.Backward()

	assert.Equal(t, []float64{
		2.4350, 0.4781, 0.2627,
		2.0619, 0.1220, 0.2241,
		3.0337, 1.3346, 0.0199,
	}, RoundFloats(result.Data.Data, 10_000))

	assert.Equal(t, 9.9721, Round(sum.Data.Data[0], 10_000))

	assert.Equal(t, []float64{
		3.4388, 1.3499, 2.4925,
		3.4388, 1.3499, 2.4925,
		3.4388, 1.3499, 2.4925,
	}, RoundFloats(input.Grad.Data, 10_000))

	assert.Equal(t, []float64{
		-25.8962, -2.6118, -1.2630,
	}, RoundFloats(weights.Grad.Data, 10_000))
}

func TestData_SubRowVector(t *testing.T) {
	input := WrapData(3, 3, 1, []float64{
		0.7081, 0.3542, 0.1054,
		0.5996, 0.0904, 0.0899,
		0.8822, 0.9887, 0.0080,
	})

	weights := WrapData(3, 1, 1, []float64{
		0.2908, 0.7408, 0.4012,
	})

	result := input.SubRowVector(weights)

	sum := result.Sum()
	sum.Backward()

	assert.Equal(t, []float64{
		0.4173, -0.3866, -0.2958,
		0.3088, -0.6504, -0.3113,
		0.5914, 0.2479, -0.3932,
	}, RoundFloats(result.Data.Data, 10_000))

	assert.Equal(t, -0.4719, Round(sum.Data.Data[0], 10_000))

	assert.Equal(t, []float64{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
	}, RoundFloats(input.Grad.Data, 10_000))

	assert.Equal(t, []float64{
		-3, -3, -3,
	}, RoundFloats(weights.Grad.Data, 10_000))
}

func TestData_MatrixMultiply(t *testing.T) {
	input := WrapData(3, 3, 1, []float64{
		0.7081, 0.3542, 0.1054,
		0.5996, 0.0904, 0.0899,
		0.8822, 0.9887, 0.0080,
	})

	weights := WrapData(1, 3, 1, []float64{
		0.2908, 0.7408, 0.4012,
	})

	result := input.MatrixMultiply(weights)

	sum := result.Sum()
	sum.Backward()

	assert.Equal(t, []float64{
		0.5106,
		0.2774,
		0.9922,
	}, RoundFloats(result.Data.Data, 10_000))

	assert.Equal(t, 1.7802, Round(sum.Data.Data[0], 10_000))

	assert.Equal(t, []float64{
		0.2908, 0.7408, 0.4012,
		0.2908, 0.7408, 0.4012,
		0.2908, 0.7408, 0.4012,
	}, RoundFloats(input.Grad.Data, 10_000))

	assert.Equal(t, []float64{
		2.1899, 1.4333, 0.2033,
	}, RoundFloats(weights.Grad.Data, 10_000))
}

func TestData_Pow(t *testing.T) {
	data := WrapData(3, 3, 1, []float64{
		0.7081, 0.3542, 0.1054,
		0.5996, 0.0904, 0.0899,
		0.8822, 0.9887, 0.0080,
	})
	std := data.Pow(2)
	std.Sum().Backward()

	assert.Equal(t, []float64{
		0.50141, 0.12546, 0.01111,
		0.35952, 0.00817, 0.00808,
		0.77828, 0.97753, 6e-05,
	}, RoundFloats(std.Data.Data, 100_000))

	assert.Equal(t, []float64{
		1.4162, 0.7084, 0.2108,
		1.1992, 0.1808, 0.1798,
		1.7644, 1.9774, 0.0160,
	}, RoundFloats(data.Grad.Data, 10_000))
}
