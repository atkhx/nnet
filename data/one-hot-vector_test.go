package data

import (
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestNewOneHotVector(t *testing.T) {
	type testCase struct {
		index          int
		count          int
		expectedVector *Data
		expectedError  error
	}

	testCases := map[string]testCase{
		"countToLow": {
			expectedVector: nil,
			expectedError:  ErrorVectorCountToLow,
		},
		"indexToLow": {
			index:          -1,
			expectedVector: nil,
			expectedError:  ErrorVectorIndexToLow,
		},
		"indexToHigh": {
			index:          1,
			count:          1,
			expectedVector: nil,
			expectedError:  ErrorVectorIndexToHigh,
		},
		"lowHotPoint": {
			index: 0,
			count: 5,
			expectedVector: &Data{
				Dims: []int{5, 1, 1},
				Data: []float64{1, 0, 0, 0, 0},
			},
		},
		"highHotPoint": {
			index: 4,
			count: 5,
			expectedVector: &Data{
				Dims: []int{5, 1, 1},
				Data: []float64{0, 0, 0, 0, 1},
			},
		},
		"middleHotPoint": {
			index: 2,
			count: 5,
			expectedVector: &Data{
				Dims: []int{5, 1, 1},
				Data: []float64{0, 0, 1, 0, 0},
			},
		},
	}

	for name, tc := range testCases {
		tc := tc
		t.Run(name, func(t *testing.T) {
			actualVector, actualError := NewOneHotVector(tc.index, tc.count)
			if tc.expectedError != nil {
				assert.Nil(t, actualVector)
				assert.Equal(t, tc.expectedError, errors.Cause(actualError))
			} else {
				assert.Equal(t, tc.expectedVector, actualVector)
				assert.NoError(t, actualError)
			}
		})
	}
}

func TestNewOneHotVectors(t *testing.T) {
	type testCase struct {
		count           int
		expectedVectors []*Data
		expectedError   error
	}

	testCases := map[string]testCase{
		"countToLow": {
			expectedVectors: nil,
			expectedError:   ErrorVectorCountToLow,
		},
		"oneVector": {
			count: 1,
			expectedVectors: []*Data{
				{
					Dims: []int{1, 1, 1},
					Data: []float64{1.0},
				},
			},
		},
		"multiVector": {
			count: 3,
			expectedVectors: []*Data{
				{
					Dims: []int{3, 1, 1},
					Data: []float64{1.0, 0.0, 0.0},
				},
				{
					Dims: []int{3, 1, 1},
					Data: []float64{0.0, 1.0, 0.0},
				},
				{
					Dims: []int{3, 1, 1},
					Data: []float64{0.0, 0.0, 1.0},
				},
			},
		},
	}

	for name, tc := range testCases {
		tc := tc
		t.Run(name, func(t *testing.T) {
			actualVector, actualError := NewOneHotVectors(tc.count)
			if tc.expectedError != nil {
				assert.Nil(t, actualVector)
				assert.Equal(t, tc.expectedError, errors.Cause(actualError))
			} else {
				assert.Equal(t, tc.expectedVectors, actualVector)
				assert.NoError(t, actualError)
			}
		})
	}
}
