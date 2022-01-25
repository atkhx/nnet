package relu

import (
	"github.com/stretchr/testify/assert"
	"math"
	"testing"
)

func TestActivation_Forward(t *testing.T) {
	type testCaseData struct {
		value    float64
		expected float64
	}

	type testCase struct {
		cases map[string]testCaseData
	}

	testCases := []testCase{
		{
			cases: map[string]testCaseData{
				"greaterThanZero": {value: 1, expected: 1},
				"equalsZero":      {value: 0.0, expected: 0.0},
				"lessThanZero":    {value: -0.5, expected: 0.0},
				"maxValue":        {value: math.MaxFloat64, expected: math.MaxFloat64},
				"minValue":        {value: -math.MaxFloat64, expected: 0.0},
			},
		},
	}

	for _, test := range testCases {
		for name, tc := range test.cases {
			tc := tc
			t.Run(name, func(t *testing.T) {
				assert.Equal(t, tc.expected, New().Forward(tc.value))
			})
		}
	}
}

func TestActivation_Backward(t *testing.T) {
	type testCaseData struct {
		value    float64
		expected float64
	}

	type testCase struct {
		cases map[string]testCaseData
	}

	testCases := []testCase{
		{
			cases: map[string]testCaseData{
				"greaterThanZero": {value: 15, expected: 1.0},
				"equalsZero":      {value: 0.0, expected: 0.0},
				"lessThanZero":    {value: -0.5, expected: 0.0},
				"maxValue":        {value: math.MaxFloat64, expected: 1},
				"minValue":        {value: -math.MaxFloat64, expected: 0.0},
			},
		},
	}

	for _, test := range testCases {
		for name, tc := range test.cases {
			tc := tc
			t.Run(name, func(t *testing.T) {
				assert.Equal(t, tc.expected, New().Backward(tc.value))
			})
		}
	}
}
