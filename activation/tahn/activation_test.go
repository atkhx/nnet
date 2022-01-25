package tahn

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestActivation_Forward(t *testing.T) {
	type testCase struct {
		value    float64
		expected float64
	}
	testCases := map[string]testCase{
		"a": {
			value:    1.0,
			expected: 0.7615941559557649,
		},
		"b": {
			value:    0.5,
			expected: 0.46211715726000974,
		},
		"c": {
			value:    0.0,
			expected: 0.0,
		},
		"d": {
			value:    -0.5,
			expected: -0.46211715726000974,
		},
		"e": {
			value:    -1.0,
			expected: -0.7615941559557649,
		},
	}

	for name, tc := range testCases {
		fn := New()
		tc := tc
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.expected, fn.Forward(tc.value))
		})
	}
}

func TestActivation_Backward(t *testing.T) {
	type testCase struct {
		value    float64
		expected float64
	}
	testCases := map[string]testCase{
		"a": {
			value:    1.0,
			expected: 0.0,
		},
		"b": {
			value:    0.5,
			expected: 0.75,
		},
		"c": {
			value:    0.0,
			expected: 1.0,
		},
		"d": {
			value:    -0.5,
			expected: 0.75,
		},
		"e": {
			value:    -1.0,
			expected: 0.0,
		},
	}

	for name, tc := range testCases {
		fn := New()
		tc := tc
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.expected, fn.Backward(tc.value))
		})
	}
}
