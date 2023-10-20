package blas32

import (
	"context"
	"math"
	"testing"

	"github.com/atkhx/nnet/num"
	"github.com/stretchr/testify/require"
)

var testDevice = &Device{}

func roundFloats(t *testing.T, src Float32s) Float32s {
	t.Helper()
	dst := NewFloat32s(len(src))
	for i, f := range src {
		dst[i] = float32(math.Round(10_000*float64(f)) / 10_000)
	}
	return dst
}

func requireEqualFloats(t *testing.T, expected, actual Float32s) {
	t.Helper()
	require.Equal(t, roundFloats(t, expected), roundFloats(t, actual))
}

func newTestData(t *testing.T, dims num.Dims, data, grad Float32s) *num.Data {
	t.Helper()
	require.Equal(t, dims.Size(), len(data))
	aData := testDevice.NewData(dims)
	Float32s(aData.Data).CopyFrom(data)
	Float32s(aData.Grad).CopyFrom(grad)
	aData.SkipResetGrad = true

	return aData
}

func runPipelines(t *testing.T, lastNode *num.Data) {
	t.Helper()
	pipeline := NewPipeline(lastNode)
	pipeline.Forward(context.Background())
	pipeline.Backward(context.Background())
}

func TestDevice_Sqrt(t *testing.T) {
	aData := newTestData(t, num.NewDims(2, 2), Float32s{4, 9, 16, 25}, Float32s{1, 1, 1, 1})
	bData := testDevice.Sqrt(aData)
	Float32s(bData.Data).Fill(15) // initialize data to check that it overrides

	runPipelines(t, bData)

	requireEqualFloats(t, Float32s{2, 3, 4, 5}, bData.Data)
	requireEqualFloats(t, Float32s{1, 1, 1, 1}, bData.Grad)
	requireEqualFloats(t, Float32s{
		1 + 0.5/2., 1 + 0.5/3.,
		1 + 0.5/4., 1 + 0.5/5.,
	}, aData.Grad)
}

func TestDevice_Mean(t *testing.T) {
	aData := newTestData(t, num.NewDims(2, 2), Float32s{1, 2, 3, 4}, Float32s{1, 1, 1, 1})
	bData := testDevice.Mean(aData)
	Float32s(bData.Data).Fill(15) // initialize data to check that it overrides

	runPipelines(t, bData)

	requireEqualFloats(t, Float32s{2.5}, bData.Data)
	requireEqualFloats(t, Float32s{1.0}, bData.Grad)
	requireEqualFloats(t, Float32s{1.25, 1.25, 1.25, 1.25}, aData.Grad)
}

func TestDevice_MeanByRows(t *testing.T) {
	aData := newTestData(t, num.NewDims(2, 2), Float32s{1, 2, 3, 4}, Float32s{1, 1, 1, 1})
	bData := testDevice.MeanByRows(aData)
	Float32s(bData.Data).Fill(15) // initialize data to check that it overrides

	runPipelines(t, bData)

	require.Equal(t, num.NewDims(1, 2), bData.Dims)

	requireEqualFloats(t, Float32s{1.5, 3.5}, bData.Data)
	requireEqualFloats(t, Float32s{1.0, 1.0}, bData.Grad)
	requireEqualFloats(t, Float32s{1.5, 1.5, 1.5, 1.5}, aData.Grad)
}

func TestDevice_VarianceByRows(t *testing.T) {
	aData := newTestData(t, num.NewDims(2, 2), Float32s{1, 2, 3, 4}, Float32s{1, 1, 1, 1})
	mData := testDevice.MeanByRows(aData)
	Float32s(mData.Data).Fill(15) // initialize data to check that it overrides

	bData := testDevice.VarianceByRows(aData, mData)
	Float32s(bData.Data).Fill(16) // initialize data to check that it overrides

	runPipelines(t, bData)

	// Check variance by rows
	require.Equal(t, num.NewDims(1, 2), bData.Dims)
	requireEqualFloats(t, Float32s{0.5, 0.5}, bData.Data)
	requireEqualFloats(t, Float32s{1.0, 1.0}, bData.Grad)

	// Check mean by rows
	require.Equal(t, num.NewDims(1, 2), mData.Dims)
	requireEqualFloats(t, Float32s{1.5, 3.5}, mData.Data)
	requireEqualFloats(t, Float32s{0.0, 0.0}, mData.Grad)

	// Check aData gradients
	requireEqualFloats(t, Float32s{0, 2, 0, 2}, aData.Grad)
}

func TestDevice_Relu(t *testing.T) {
	aData := newTestData(t, num.NewDims(2, 2), Float32s{1, 0.1, -0.2, 0}, Float32s{1, 1, 1, 1})
	bData := testDevice.Relu(aData)
	Float32s(bData.Data).Fill(15) // initialize data to check that it overrides

	runPipelines(t, bData)

	require.Equal(t, num.NewDims(2, 2), bData.Dims)
	requireEqualFloats(t, Float32s{1, 0.1, 0, 0}, bData.Data)
	requireEqualFloats(t, Float32s{1, 1, 1, 1}, bData.Grad)

	requireEqualFloats(t, Float32s{2, 2, 1, 1}, aData.Grad)
}

func TestDevice_AddScalar(t *testing.T) {
	aData := newTestData(t, num.NewDims(2, 2), Float32s{1, 2, 3, 4}, Float32s{1, 1, 1, 1})
	bData := testDevice.AddScalar(aData, 5)
	Float32s(bData.Data).Fill(15) // initialize data to check that it overrides

	runPipelines(t, bData)

	require.Equal(t, num.NewDims(2, 2), bData.Dims)
	requireEqualFloats(t, Float32s{6, 7, 8, 9}, bData.Data)
	requireEqualFloats(t, Float32s{1, 1, 1, 1}, bData.Grad)
	requireEqualFloats(t, Float32s{2, 2, 2, 2}, aData.Grad)
}

func TestDevice_MulScalar(t *testing.T) {
	aData := newTestData(t, num.NewDims(2, 2), Float32s{1, 2, 3, 4}, Float32s{1, 1, 1, 1})
	bData := testDevice.MulScalar(aData, 3)
	Float32s(bData.Data).Fill(15) // initialize data to check that it overrides

	runPipelines(t, bData)

	require.Equal(t, num.NewDims(2, 2), bData.Dims)
	requireEqualFloats(t, Float32s{3, 6, 9, 12}, bData.Data)
	requireEqualFloats(t, Float32s{1, 1, 1, 1}, bData.Grad)
	requireEqualFloats(t, Float32s{4, 4, 4, 4}, aData.Grad)
}

func TestDevice_Add(t *testing.T) {
	type testCase struct {
		aDims num.Dims
		aData Float32s
		aGrad Float32s

		bDims num.Dims
		bData Float32s
		bGrad Float32s

		expectedCDims num.Dims
		expectedCData Float32s
		expectedCGrad Float32s
		expectedAGrad Float32s
		expectedBGrad Float32s
	}

	testCases := map[string]testCase{
		"3d plus 3d": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(2, 2, 2),
			bData: Float32s{
				2, 3,
				4, 5,

				3, 4,
				5, 6,
			},
			bGrad:         Float32s{2, 2, 2, 2, 2, 2, 2, 2},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				3, 5,
				7, 9,

				5, 7,
				9, 11,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{2, 2, 2, 2, 2, 2, 2, 2},
			expectedBGrad: Float32s{3, 3, 3, 3, 3, 3, 3, 3},
		},
		"3d plus 2d": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(2, 2),
			bData: Float32s{
				2, 3,
				4, 5,
			},
			bGrad:         Float32s{2, 2, 2, 2},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				3, 5,
				7, 9,

				4, 6,
				8, 10,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{2, 2, 2, 2, 2, 2, 2, 2},
			expectedBGrad: Float32s{4, 4, 4, 4},
		},
		"3d plus row": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(2),
			bData: Float32s{
				2, 3,
			},
			bGrad:         Float32s{2, 2},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				3, 5,
				5, 7,

				4, 6,
				6, 8,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{2, 2, 2, 2, 2, 2, 2, 2},
			expectedBGrad: Float32s{6, 6},
		},
		"3d plus column": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(1, 2),
			bData: Float32s{
				2,
				3,
			},
			bGrad:         Float32s{2, 2},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				3, 4,
				6, 7,

				4, 5,
				7, 8,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{2, 2, 2, 2, 2, 2, 2, 2},
			expectedBGrad: Float32s{6, 6},
		},
		"3d plus point": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(1),
			bData: Float32s{
				2,
			},
			bGrad:         Float32s{2},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				3, 4,
				5, 6,

				4, 5,
				6, 7,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{2, 2, 2, 2, 2, 2, 2, 2},
			expectedBGrad: Float32s{10},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			aData := newTestData(t, tc.aDims, tc.aData, tc.aGrad)
			bData := newTestData(t, tc.bDims, tc.bData, tc.bGrad)
			cData := testDevice.Add(aData, bData)
			Float32s(cData.Data).Fill(15) // initialize data to check that it overrides

			runPipelines(t, cData)

			require.Equal(t, tc.expectedCDims, cData.Dims)
			requireEqualFloats(t, tc.expectedCData, cData.Data)
			requireEqualFloats(t, tc.expectedCGrad, cData.Grad)

			requireEqualFloats(t, tc.expectedAGrad, aData.Grad)
			requireEqualFloats(t, tc.expectedBGrad, bData.Grad)
		})
	}
}

func TestDevice_Sub(t *testing.T) {
	type testCase struct {
		aDims num.Dims
		aData Float32s
		aGrad Float32s

		bDims num.Dims
		bData Float32s
		bGrad Float32s

		expectedCDims num.Dims
		expectedCData Float32s
		expectedCGrad Float32s
		expectedAGrad Float32s
		expectedBGrad Float32s
	}

	testCases := map[string]testCase{
		"3d minus 3d": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(2, 2, 2),
			bData: Float32s{
				2, 3,
				4, 5,

				3, 4,
				5, 6,
			},
			bGrad:         Float32s{6, 6, 6, 6, 6, 6, 6, 6},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				-1, -1,
				-1, -1,

				-1, -1,
				-1, -1,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{2, 2, 2, 2, 2, 2, 2, 2},
			expectedBGrad: Float32s{5, 5, 5, 5, 5, 5, 5, 5},
		},
		"3d minus 2d": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(2, 2),
			bData: Float32s{
				2, 3,
				4, 5,
			},
			bGrad:         Float32s{7, 7, 7, 7},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				-1, -1,
				-1, -1,

				0, 0,
				0, 0,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{2, 2, 2, 2, 2, 2, 2, 2},
			expectedBGrad: Float32s{5, 5, 5, 5},
		},
		"3d minus row": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(2),
			bData: Float32s{
				2, 3,
			},
			bGrad:         Float32s{7, 7},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				-1, -1,
				1, 1,

				0, 0,
				2, 2,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{2, 2, 2, 2, 2, 2, 2, 2},
			expectedBGrad: Float32s{3, 3},
		},
		"3d minus column": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(1, 2),
			bData: Float32s{
				2,
				3,
			},
			bGrad:         Float32s{7, 7},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				-1, 0,
				0, 1,

				0, 1,
				1, 2,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{2, 2, 2, 2, 2, 2, 2, 2},
			expectedBGrad: Float32s{3, 3},
		},
		"3d minus point": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(1),
			bData: Float32s{
				2,
			},
			bGrad:         Float32s{7},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				-1, 0,
				1, 2,

				0, 1,
				2, 3,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{2, 2, 2, 2, 2, 2, 2, 2},
			expectedBGrad: Float32s{-1},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			aData := newTestData(t, tc.aDims, tc.aData, tc.aGrad)
			bData := newTestData(t, tc.bDims, tc.bData, tc.bGrad)
			cData := testDevice.Sub(aData, bData)
			Float32s(cData.Data).Fill(15) // initialize data to check that it overrides

			runPipelines(t, cData)

			require.Equal(t, tc.expectedCDims, cData.Dims)
			requireEqualFloats(t, tc.expectedCData, cData.Data)
			requireEqualFloats(t, tc.expectedCGrad, cData.Grad)

			requireEqualFloats(t, tc.expectedAGrad, aData.Grad)
			requireEqualFloats(t, tc.expectedBGrad, bData.Grad)
		})
	}
}

func TestDevice_Mul(t *testing.T) {
	type testCase struct {
		aDims num.Dims
		aData Float32s
		aGrad Float32s

		bDims num.Dims
		bData Float32s
		bGrad Float32s

		expectedCDims num.Dims
		expectedCData Float32s
		expectedCGrad Float32s
		expectedAGrad Float32s
		expectedBGrad Float32s
	}

	testCases := map[string]testCase{
		"3d mul 3d": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(2, 2, 2),
			bData: Float32s{
				2, 3,
				4, 5,

				3, 4,
				5, 6,
			},
			bGrad:         Float32s{6, 6, 6, 6, 6, 6, 6, 6},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				2, 6,
				12, 20,

				6, 12,
				20, 30,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{3, 4, 5, 6, 4, 5, 6, 7},    // [1] + bData * 1
			expectedBGrad: Float32s{7, 8, 9, 10, 8, 9, 10, 11}, // [6] + aData * 1
		},
		"3d mul 2d": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(2, 2),
			bData: Float32s{
				2, 3,
				4, 5,
			},
			bGrad:         Float32s{7, 7, 7, 7},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				2, 6,
				12, 20,

				4, 9,
				16, 25,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{3, 4, 5, 6, 3, 4, 5, 6},
			expectedBGrad: Float32s{10, 12, 14, 16},
		},
		"3d mul row": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(2),
			bData: Float32s{
				2, 3,
			},
			bGrad:         Float32s{7, 7},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				2, 6,
				6, 12,

				4, 9,
				8, 15,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{3, 4, 3, 4, 3, 4, 3, 4},
			expectedBGrad: Float32s{17, 21},
		},
		"3d mul column": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(1, 2),
			bData: Float32s{
				2,
				3,
			},
			bGrad:         Float32s{7, 7},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				2, 4,
				9, 12,

				4, 6,
				12, 15,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{3, 3, 4, 4, 3, 3, 4, 4},
			expectedBGrad: Float32s{15, 23},
		},
		"3d mul point": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(1),
			bData: Float32s{
				2,
			},
			bGrad:         Float32s{7},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				2, 4,
				6, 8,

				4, 6,
				8, 10,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{3, 3, 3, 3, 3, 3, 3, 3},
			expectedBGrad: Float32s{31},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			aData := newTestData(t, tc.aDims, tc.aData, tc.aGrad)
			bData := newTestData(t, tc.bDims, tc.bData, tc.bGrad)
			cData := testDevice.Mul(aData, bData)
			Float32s(cData.Data).Fill(15) // initialize data to check that it overrides

			runPipelines(t, cData)

			require.Equal(t, tc.expectedCDims, cData.Dims)
			requireEqualFloats(t, tc.expectedCData, cData.Data)
			requireEqualFloats(t, tc.expectedCGrad, cData.Grad)

			requireEqualFloats(t, tc.expectedAGrad, aData.Grad)
			requireEqualFloats(t, tc.expectedBGrad, bData.Grad)
		})
	}
}

func TestDevice_Div(t *testing.T) {
	type testCase struct {
		aDims num.Dims
		aData Float32s
		aGrad Float32s

		bDims num.Dims
		bData Float32s
		bGrad Float32s

		expectedCDims num.Dims
		expectedCData Float32s
		expectedCGrad Float32s
		expectedAGrad Float32s
		expectedBGrad Float32s
	}

	testCases := map[string]testCase{
		"3d div 3d": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				2, 6,
				12, 20,

				6, 12,
				20, 30,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(2, 2, 2),
			bData: Float32s{
				2, 3,
				4, 5,

				3, 4,
				5, 6,
			},
			bGrad:         Float32s{6, 6, 6, 6, 6, 6, 6, 6},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{1.5, 1.3333, 1.25, 1.2, 1.3333, 1.25, 1.2, 1.1667},
			expectedBGrad: Float32s{5.5, 5.3333, 5.25, 5.2, 5.3333, 5.25, 5.2, 5.1667},
		},
		"3d div 2d": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				2, 6,
				12, 20,

				4, 9,
				16, 25,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(2, 2),
			bData: Float32s{
				2, 3,
				4, 5,
			},
			bGrad:         Float32s{7, 7, 7, 7},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{1.5, 1.3333, 1.25, 1.2, 1.5, 1.3333, 1.25, 1.2},
			expectedBGrad: Float32s{5.5, 5.3333, 5.25, 5.2},
		},
		"3d div row": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				2, 6,
				6, 12,

				4, 9,
				8, 15,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(2),
			bData: Float32s{
				2, 3,
			},
			bGrad:         Float32s{7, 7},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{1.5, 1.3333, 1.5, 1.3333, 1.5, 1.3333, 1.5, 1.3333},
			expectedBGrad: Float32s{2, 2.3333},
		},
		"3d div column": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				2, 4,
				9, 12,

				4, 6,
				12, 15,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(1, 2),
			bData: Float32s{
				2,
				3,
			},
			bGrad:         Float32s{7, 7},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{1.5, 1.5, 1.3333, 1.3333, 1.5, 1.5, 1.3333, 1.3333},
			expectedBGrad: Float32s{3, 1.6667},
		},
		"3d div point": {
			aDims: num.NewDims(2, 2, 2),
			aData: Float32s{
				2, 4,
				6, 8,

				4, 6,
				8, 10,
			},
			aGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			bDims: num.NewDims(1),
			bData: Float32s{
				2,
			},
			bGrad:         Float32s{7},
			expectedCDims: num.NewDims(2, 2, 2),
			expectedCData: Float32s{
				1, 2,
				3, 4,

				2, 3,
				4, 5,
			},
			expectedCGrad: Float32s{1, 1, 1, 1, 1, 1, 1, 1},
			expectedAGrad: Float32s{1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5},
			expectedBGrad: Float32s{-5},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			aData := newTestData(t, tc.aDims, tc.aData, tc.aGrad)
			bData := newTestData(t, tc.bDims, tc.bData, tc.bGrad)
			cData := testDevice.Div(aData, bData)
			Float32s(cData.Data).Fill(15) // initialize data to check that it overrides

			runPipelines(t, cData)

			require.Equal(t, tc.expectedCDims, cData.Dims)
			requireEqualFloats(t, tc.expectedCData, cData.Data)
			requireEqualFloats(t, tc.expectedCGrad, cData.Grad)

			requireEqualFloats(t, tc.expectedAGrad, aData.Grad)
			requireEqualFloats(t, tc.expectedBGrad, bData.Grad)
		})
	}
}

func TestDevice_ConcatByRows(t *testing.T) {
	aData := newTestData(t, num.NewDims(2, 2), Float32s{
		1, 2,
		3, 4,
	}, Float32s{1, 1, 1, 1})

	bData := newTestData(t, num.NewDims(2, 2), Float32s{
		5, 6,
		7, 8,
	}, Float32s{2, 2, 2, 2})

	cData := testDevice.ConcatByRows(aData, bData)
	Float32s(cData.Data).Fill(15) // initialize data to check that it overrides

	runPipelines(t, cData)

	require.Equal(t, num.NewDims(4, 2), cData.Dims)
	requireEqualFloats(t, Float32s{
		1, 2, 5, 6,
		3, 4, 7, 8,
	}, cData.Data)

	requireEqualFloats(t, Float32s{
		1, 1, 1, 1,
		1, 1, 1, 1,
	}, cData.Grad)

	requireEqualFloats(t, Float32s{2, 2, 2, 2}, aData.Grad)
	requireEqualFloats(t, Float32s{3, 3, 3, 3}, bData.Grad)
}

func TestDevice_Dropout(t *testing.T) {
	randGenerator.Seed(1)

	aData := newTestData(t, num.NewDims(2, 2), Float32s{1, 2, 3, 4}, Float32s{1, 1, 1, 1})
	bData := testDevice.Dropout(aData, 0.5)
	Float32s(bData.Data).Fill(15) // initialize data to check that it overrides

	runPipelines(t, bData)

	require.Equal(t, num.NewDims(2, 2), bData.Dims)
	requireEqualFloats(t, Float32s{1, 2, 3, 0}, bData.Data)
	requireEqualFloats(t, Float32s{1, 1, 1, 1}, bData.Grad)
	requireEqualFloats(t, Float32s{2, 2, 2, 1}, aData.Grad)
}

func TestDevice_Reshape(t *testing.T) {
	aData := newTestData(t, num.NewDims(3, 2), Float32s{1, 2, 3, 4, 5, 6}, Float32s{3, 3, 3, 3, 3, 3})
	bData := testDevice.Reshape(aData, num.NewDims(2, 3))
	cData := testDevice.Reshape(bData, num.NewDims(6, 1))
	mData := testDevice.Mean(cData)

	runPipelines(t, mData)

	require.Equal(t, num.NewDims(1, 1), mData.Dims)
	require.Equal(t, num.NewDims(6, 1), cData.Dims)
	require.Equal(t, num.NewDims(2, 3), bData.Dims)
	require.Equal(t, num.NewDims(3, 2), aData.Dims)

	expectedData := Float32s{1, 2, 3, 4, 5, 6}

	requireEqualFloats(t, Float32s{21. / 6.}, mData.Data)
	requireEqualFloats(t, expectedData, cData.Data)
	requireEqualFloats(t, expectedData, bData.Data)
	requireEqualFloats(t, expectedData, aData.Data)

	mGrad := float32(1. / 6.)
	rGrad := 3. + mGrad

	expectedGrad := Float32s{rGrad, rGrad, rGrad, rGrad, rGrad, rGrad}

	requireEqualFloats(t, Float32s{1}, mData.Grad)
	requireEqualFloats(t, expectedGrad, cData.Grad)
	requireEqualFloats(t, expectedGrad, bData.Grad)
	requireEqualFloats(t, expectedGrad, aData.Grad)

	// Check that "Data" slice is common for reshape results
	expectedData = Float32s{1, 2, 7, 7, 5, 6}
	cData.Data[2] = 7
	cData.Data[3] = 7

	requireEqualFloats(t, expectedData, cData.Data)
	requireEqualFloats(t, expectedData, bData.Data)
	requireEqualFloats(t, expectedData, aData.Data)

	// Check that "Grad" slice is common for reshape results

	expectedGrad = Float32s{rGrad, rGrad, 9, 9, rGrad, rGrad}
	cData.Grad[2] = 9
	cData.Grad[3] = 9

	requireEqualFloats(t, expectedGrad, cData.Grad)
	requireEqualFloats(t, expectedGrad, bData.Grad)
	requireEqualFloats(t, expectedGrad, aData.Grad)
}

func TestDevice_CrossEntropyPos(t *testing.T) {
	aData := newTestData(t, num.NewDims(3, 2), Float32s{
		0, 1, 0, // softmax: 0.21194157 0.5761169 0.21194157
		2, 3, 1, // softmax: 0.24472848 0.66524094 0.09003057
	}, Float32s{3, 3, 3, 3, 3, 3})

	targets := newTestData(t, num.NewDims(1, 2), Float32s{
		1,
		2,
	}, Float32s{1, 1})

	bData := testDevice.CrossEntropyPos(aData, targets)

	runPipelines(t, bData)

	require.Equal(t, num.NewDims(1, 2), bData.Dims)
	requireEqualFloats(t, Float32s{1, 1}, bData.Grad)
	requireEqualFloats(t, Float32s{
		float32(-math.Log(0.5761169)),
		float32(-math.Log(0.09003057)),
	}, bData.Data)

	requireEqualFloats(t, Float32s{
		3 + 1*0.21194157, 3 + 1*(0.5761169-1), 3 + 1*0.21194157,
		3 + 1*0.24472848, 3 + 1*0.66524094, 3 + 1*(0.09003057-1),
	}, aData.Grad)
}

func TestDevice_Embeddings(t *testing.T) {
	alphabetSize := 2
	featuresCount := 3
	contextSize := 4
	batchSize := 2

	tokenEmbeddings := newTestData(t, num.NewDims(featuresCount, alphabetSize), Float32s{
		1, 2, 3, // features for token 0
		4, 5, 6, // features for token 1
	}, Float32s{3, 3, 3, 3, 3, 3})

	positionsEmbeddings := newTestData(t, num.NewDims(featuresCount, contextSize), Float32s{
		1, 2, 3, // features for position 0
		2, 3, 4, // features for position 1
		3, 4, 5, // features for position 2
		4, 5, 6, // features for position 3
	}, Float32s{
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
	})

	aData := newTestData(t, num.NewDims(contextSize, batchSize), Float32s{
		0, 1, 1, 0, // phrase 1
		1, 0, 0, 0, // phrase 2
	}, Float32s{3, 3, 3, 3, 3, 3})

	bData := testDevice.Embeddings(aData, tokenEmbeddings, positionsEmbeddings)
	Float32s(bData.Grad).Fill(17)

	runPipelines(t, bData)

	require.Equal(t, num.NewDims(featuresCount, contextSize, batchSize), bData.Dims)
	requireEqualFloats(t, Float32s{
		// phrase 1
		2, 4, 6, // f[i] + p[i] = [1, 2, 3] add [1, 2, 3]
		6, 8, 10, // f[i] + p[i] = [4, 5, 6] add [2, 3, 4]
		7, 9, 11, // f[i] + p[i] = [4, 5, 6] add [3, 4, 5]
		5, 7, 9, // f[i] + p[i] = [1, 2, 3] add [4, 5, 6]

		// phrase 2
		5, 7, 9, // f[i] + p[i] = [4, 5, 6] add [1, 2, 3]
		3, 5, 7, // f[i] + p[i] = [1, 2, 3] add [2, 3, 4]
		4, 6, 8, // f[i] + p[i] = [1, 2, 3] add [3, 4, 5]
		5, 7, 9, // f[i] + p[i] = [1, 2, 3] add [4, 5, 6]
	}, bData.Data)
	requireEqualFloats(t, Float32s{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,

		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
	}, bData.Grad)

	requireEqualFloats(t, Float32s{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, positionsEmbeddings.Grad)
	requireEqualFloats(t, Float32s{8, 8, 8, 6, 6, 6}, tokenEmbeddings.Grad)
}

func TestDevice_Transpose(t *testing.T) {
	aData := newTestData(t, num.NewDims(2, 3, 2), Float32s{
		1, 2,
		3, 4,
		5, 6,

		2, 3,
		4, 5,
		6, 7,
	}, Float32s{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})

	bData := testDevice.Transpose(aData)
	Float32s(bData.Grad).CopyFrom(Float32s{1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4})
	bData.SkipResetGrad = true

	mData := testDevice.Mean(bData)

	runPipelines(t, mData)

	require.Equal(t, num.NewDims(1), mData.Dims)
	requireEqualFloats(t, Float32s{4}, mData.Data)
	requireEqualFloats(t, Float32s{1}, mData.Grad)

	require.Equal(t, num.NewDims(3, 2, 2), bData.Dims)
	requireEqualFloats(t, Float32s{
		1, 3, 5,
		2, 4, 6,

		2, 4, 6,
		3, 5, 7,
	}, bData.Data)

	requireEqualFloats(t, Float32s{
		1.0833, 1.0833, 1.0833,
		2.0833, 2.0833, 2.0833,

		3.0833, 3.0833, 3.0833,
		4.0833, 4.0833, 4.0833,
	}, bData.Grad)

	requireEqualFloats(t, Float32s{
		2.0833, 3.0833,
		2.0833, 3.0833,
		2.0833, 3.0833,

		4.0833, 5.0833,
		4.0833, 5.0833,
		4.0833, 5.0833,
	}, aData.Grad)
}

func TestDevice_TriangleLowerSoftmax(t *testing.T) {
	aData := newTestData(t, num.NewDims(3, 2, 2), Float32s{
		1, 2, 3,
		4, 5, 6,

		3, 4, 5,
		1, 8, 1,
	}, Float32s{
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
	})

	bData := testDevice.TriangleLowerSoftmax(aData)

	runPipelines(t, bData)

	require.Equal(t, num.NewDims(3, 2, 2), bData.Dims)
	requireEqualFloats(t, Float32s{
		1.0, 0.0, 0.0,
		0.2689, 0.7311, 0.0,

		1.0, 0.0, 0.0,
		0.0009, 0.9991, 0.0,
	}, bData.Data)

	requireEqualFloats(t, Float32s{
		1, 1, 1,
		1, 1, 1,

		1, 1, 1,
		1, 1, 1,
	}, bData.Grad)

	requireEqualFloats(t, Float32s{
		0, 0, 0,
		0, 0, 0,

		0, 0, 0,
		0, 0, 0,
	}, aData.Grad)
}

func TestDevice_MatrixMultiply2D(t *testing.T) {
	aData := newTestData(t, num.NewDims(3, 2, 1), Float32s{
		1, 2, 3,
		4, 5, 6,
	}, Float32s{2, 2, 2, 2, 2, 2})

	bData := newTestData(t, num.NewDims(2, 3, 1), Float32s{
		1, 0,
		0, 1,
		1, 0,
	}, Float32s{1, 1, 1, 1, 1, 1})

	cData := testDevice.MatrixMultiply2D(aData, bData, 1)

	runPipelines(t, cData)

	require.Equal(t, num.NewDims(2, 2, 1), cData.Dims)
	requireEqualFloats(t, Float32s{
		4, 2,
		10, 5,
	}, cData.Data)

	requireEqualFloats(t, Float32s{
		1, 1,
		1, 1,
	}, cData.Grad)

	requireEqualFloats(t, Float32s{
		3, 3, 3,
		3, 3, 3,
	}, aData.Grad)

	requireEqualFloats(t, Float32s{
		6, 6,
		8, 8,
		10, 10,
	}, bData.Grad)
}
