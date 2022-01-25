package sigmoid

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
			value:    0.7,
			expected: 0.6681877721681662,
		},
		"b": {
			value:    1.0,
			expected: 0.7310585786300049,
		},
		"c": {
			value:    0.0,
			expected: 0.5,
		},
		"d": {
			value:    -0.7,
			expected: 0.3318122278318339,
		},
		"e": {
			value:    -1.0,
			expected: 0.2689414213699951,
		},
		"f": {
			value:    9,
			expected: 0.9998766054240137,
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
			value:    0.0,
			expected: 0.0,
		},
		"b": {
			value:    1.0,
			expected: 0.0,
		},
		"c": {
			value:    0.3,
			expected: 0.21,
		},
		"d": {
			value:    -0.7,
			expected: -1.19,
		},
		"e": {
			value:    -1.0,
			expected: -2.0,
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
