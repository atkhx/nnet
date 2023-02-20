package loss

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/floats"
)

func TestClassification_GetGradient(t *testing.T) {
	type testCase struct {
		target   *data.Data
		output   *data.Data
		expected *data.Data
	}
	testCases := map[string]testCase{
		"PositiveOutput": {
			target: data.NewVector(0.0, 1.0, 0.0),
			output: data.NewVector(0.5, 0.6, 0.3),
			// -(t / o) + ((1 - t) / (1 - o))
			expected: data.NewVector(
				1/0.5,    // -0.0/0.5+((1-0.0)/(1-0.5)),
				-1.0/0.6, // -1.0/0.6+((1-1)/(1-0.6)),
				1/0.7,    // -0.0/0.3+((1-0)/(1-0.3)),
			),
		},
		"NegativeOutput": {
			target: data.NewVector(0.0, 1.0, 0.0),
			output: data.NewVector(1.4, -0.7, 0.3),
			// -(t / o) + ((1 - t) / (1 - o))
			expected: data.NewVector(
				-1/0.4, // -0.0/1.4+((1-0.0)/(1-1.4)),
				1/0.7,  // -1.0/-0.7+((1-1.0)/(1+0.7)),
				1/0.7,  // -0.0/0.3+((1-0.0)/(1-0.3)),
			),
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			actual := NewClassificationLossFunc(tc.target)(tc.output).GetGradient()

			floats.Round(tc.expected.Data, 10000)
			floats.Round(actual.Data, 10000)
			assert.Equal(t, tc.expected, actual)
		})
	}
}

func TestClassification_GetError(t *testing.T) {
	type testCase struct {
		target   *data.Data
		output   *data.Data
		expected float64
	}

	testCases := map[string]testCase{
		"predictionFailed": {
			target:   data.NewVector(0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
			output:   data.NewVector(0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
			expected: -math.Log(minimalNonZeroFloat),
		},
		"predictionAbsSuccess": {
			target:   data.NewVector(0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
			output:   data.NewVector(0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
			expected: 0,
		},
		"predictionPartlySuccess": {
			target:   data.NewVector(0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
			output:   data.NewVector(0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3),
			expected: -math.Log(0.3),
		},
	}

	for name, tc := range testCases {
		tc := tc
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.expected, NewClassificationLossFunc(tc.target)(tc.output).GetError())
		})
	}
}
