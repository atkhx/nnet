package data

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestData_Rotate180(t *testing.T) {
	type testCase struct {
		Source   *Data
		Expected *Data
	}

	testCases := map[string]testCase{
		"empty": {
			Source:   &Data{},
			Expected: &Data{Dims: []int{}, Data: []float64{}},
		},
		"one element": {
			Source:   &Data{Dims: []int{1}, Data: []float64{7.5}},
			Expected: &Data{Dims: []int{1}, Data: []float64{7.5}},
		},
		"two element 2d": {
			Source: &Data{
				Dims: []int{2, 2},
				Data: []float64{
					1, 2,
					3, 4,
				},
			},
			Expected: &Data{
				Dims: []int{2, 2},
				Data: []float64{
					4, 3,
					2, 1,
				},
			},
		},
		"tree element 3d": {
			Source: &Data{
				Dims: []int{3, 3, 3},
				Data: []float64{
					1, 2, 3,
					4, 5, 6,
					7, 8, 9,

					11, 12, 13,
					14, 15, 16,
					17, 18, 19,

					21, 22, 23,
					24, 25, 26,
					27, 28, 29,
				},
			},
			Expected: &Data{
				Dims: []int{3, 3, 3},
				Data: []float64{
					9, 8, 7,
					6, 5, 4,
					3, 2, 1,

					19, 18, 17,
					16, 15, 14,
					13, 12, 11,

					29, 28, 27,
					26, 25, 24,
					23, 22, 21,
				},
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.Expected, tc.Source.Rotate180())
		})
	}
}
