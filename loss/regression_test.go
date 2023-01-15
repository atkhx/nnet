package loss

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/atkhx/nnet/data"
)

func TestRegression_GetDeltas(t *testing.T) {
	loss := NewRegression()

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
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.expected, loss.GetDeltas(tc.target, tc.output))
		})
	}
}

func TestRegression_GetError(t *testing.T) {
	loss := NewRegression()

	type testCase struct {
		target   []float64
		result   []float64
		expected float64
	}
	testCases := map[string]testCase{
		"a": {
			target:   []float64{0.5, 1.0, 0.7},
			result:   []float64{0.5, 0.7, 0.3},
			expected: 0.125,
		},
		"b": {
			target:   []float64{0.0, 0.0, 1.0},
			result:   []float64{0.5, 0.7, 0.3},
			expected: 0.615,
		},
		"c": {
			target:   []float64{1.0, 0.2, 0.1},
			result:   []float64{0.5, 0.7, 0.3},
			expected: 0.26999999999999996,
		},
	}

	for name, tc := range testCases {
		tc := tc
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.expected, loss.GetError(tc.target, tc.result))
		})
	}
}
