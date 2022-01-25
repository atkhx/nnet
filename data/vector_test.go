package data

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestNewVector(t *testing.T) {
	type testCase struct {
		values   []float64
		expected *Data
	}

	testCases := map[string]testCase{
		"zeroLengthVector": {
			expected: &Data{
				Dims: []int{0, 1, 1},
				Data: []float64{},
			},
		},
		"oneLengthVector": {
			values: []float64{0.17},
			expected: &Data{
				Dims: []int{1, 1, 1},
				Data: []float64{0.17},
			},
		},
		"multiLengthVector": {
			values: []float64{0.17, 1.0, -0.15, 0.89},
			expected: &Data{
				Dims: []int{4, 1, 1},
				Data: []float64{0.17, 1.0, -0.15, 0.89},
			},
		},
	}

	for name, tc := range testCases {
		tc := tc
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.expected, NewVector(tc.values...))
		})
	}
}

func TestNewVectorNotLinkToSource(t *testing.T) {
	source := []float64{1, 2, 3, 4, 5}
	vector := NewVector(source...)

	expected := []float64{1, 2, 3, 4, 5}
	assert.EqualValues(t, expected, vector.Data, "invalid values in vector")

	// change source slice values
	for i := 0; i < len(source); i++ {
		source[i] = float64(len(source) - i)
	}

	// check than values in vector not linked to then original source slice
	assert.EqualValues(t, expected, vector.Data, "values changed with source")
}
