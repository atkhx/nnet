package classification

import (
	"github.com/atkhx/nnet/data"
	"github.com/stretchr/testify/assert"
	"math"
	"testing"
)

func TestLoss_GetDeltas(t *testing.T) {
	loss := New()

	type testCase struct {
		target   *data.Data
		output   *data.Data
		expected *data.Data
	}
	testCases := map[string]testCase{
		"PositiveOutput": {
			target:   data.NewVector(0.0, 1.0, 0.0),
			output:   data.NewVector(0.5, 0.7, 0.3),
			expected: data.NewVector(0.5, -0.30000000000000004, 0.3),
		},
		"NegativeOutput": {
			target:   data.NewVector(0.0, 1.0, 0.0),
			output:   data.NewVector(1.4, -0.7, 0.3),
			expected: data.NewVector(1.4, -1.7, 0.3),
		},
	}

	for name, tc := range testCases {
		tc := tc
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.expected, loss.GetDeltas(tc.target, tc.output))
		})
	}
}

func TestLoss_GetError(t *testing.T) {
	type testCase struct {
		target   []float64
		output   []float64
		expected float64
	}

	testCases := map[string]testCase{
		"predictionFailed": {
			target:   []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
			output:   []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
			expected: -math.Log(minimalNonZeroFloat),
		},
		"predictionAbsSuccess": {
			target:   []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
			output:   []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
			expected: 0,
		},
		"predictionPartlySuccess": {
			target:   []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
			output:   []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3},
			expected: -math.Log(0.3),
		},
		"zeroLengths": {
			target:   []float64{},
			output:   []float64{},
			expected: 0,
		},
	}

	loss := New()

	for name, tc := range testCases {
		tc := tc
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.expected, loss.GetError(tc.target, tc.output))
		})
	}
}
