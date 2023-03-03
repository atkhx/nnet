package data

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMatrixTranspose(t *testing.T) {
	type testCase struct {
		aColCount, aRowCount int
		a                    []float64
		rColCount, rRowCount int
		r                    []float64
	}

	testCases := map[string]testCase{
		"1x1": {
			aColCount: 1,
			aRowCount: 1,
			a:         []float64{17},
			rColCount: 1,
			rRowCount: 1,
			r:         []float64{17},
		},
		"2x1": {
			aColCount: 2,
			aRowCount: 1,
			a:         []float64{17, 18},
			rColCount: 1,
			rRowCount: 2,
			r:         []float64{17, 18},
		},
		"10x2": {
			aColCount: 10,
			aRowCount: 2,
			a: []float64{
				1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
				2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
			},
			rColCount: 2,
			rRowCount: 10,
			r: []float64{
				1, 2,
				2, 3,
				3, 4,
				4, 5,
				5, 6,
				6, 7,
				7, 8,
				8, 9,
				9, 0,
				0, 11,
			},
		},
		"2x3": {
			aColCount: 2,
			aRowCount: 3,
			a: []float64{
				17, 18,
				19, 20,
				21, 22,
			},
			rColCount: 3,
			rRowCount: 2,
			r: []float64{
				17, 19, 21,
				18, 20, 22,
			},
		},
		"3x2": {
			aColCount: 3,
			aRowCount: 2,
			a: []float64{
				17, 18, 19,
				20, 21, 22,
			},
			rColCount: 2,
			rRowCount: 3,
			r: []float64{
				17, 20,
				18, 21,
				19, 22,
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			actColCount, actRowCount, actData := MatrixTranspose(tc.aColCount, tc.aRowCount, tc.a)

			assert.Equal(t, tc.rColCount, actColCount)
			assert.Equal(t, tc.rRowCount, actRowCount)
			assert.Equal(t, tc.r, actData)
		})
	}
}

func TestMatrixAddRowVector(t *testing.T) {
	type testCase struct {
		aColsCount int
		aRowsCount int

		matrix []float64
		vector []float64

		expected []float64
	}

	testCases := map[string]testCase{
		"1x1 + 1": {
			aColsCount: 1,
			aRowsCount: 1,
			matrix:     []float64{17},
			vector:     []float64{1},
			expected:   []float64{18},
		},
		"1x2 + 2": {
			aColsCount: 1,
			aRowsCount: 2,
			matrix: []float64{
				17,
				18,
			},
			vector: []float64{0.5},
			expected: []float64{
				17.5,
				18.5,
			},
		},
		"3x2 + 3": {
			aColsCount: 3,
			aRowsCount: 2,
			matrix: []float64{
				1, 2, 3,
				4, 5, 6,
			},
			vector: []float64{0.5, 0.6, 0.7},
			expected: []float64{
				1.5, 2.6, 3.7,
				4.5, 5.6, 6.7,
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.expected, MatrixAddRowVector(tc.aColsCount, tc.aRowsCount, tc.matrix, tc.vector))
		})
	}
}

func TestMatrixRotate180(t *testing.T) {
	type testCase struct {
		iw, ih int
		a      []float64
		ow, oh int
		b      []float64
	}
	testCases := map[string]testCase{
		"1x1": {
			iw: 1,
			ih: 1,
			a:  []float64{1},
			ow: 1,
			oh: 1,
			b:  []float64{1},
		},
		"2x2": {
			iw: 2,
			ih: 2,
			a: []float64{
				1, 2,
				3, 4,
			},
			ow: 2,
			oh: 2,
			b: []float64{
				4, 3,
				2, 1,
			},
		},
		"3x3": {
			iw: 3,
			ih: 3,
			a: []float64{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
			},
			ow: 3,
			oh: 3,
			b: []float64{
				9, 8, 7,
				6, 5, 4,
				3, 2, 1,
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			actualW, actualH, actualB := MatrixRotate180(tc.iw, tc.ih, tc.a)
			assert.Equal(t, tc.ow, actualW)
			assert.Equal(t, tc.oh, actualH)
			assert.Equal(t, tc.b, actualB)
		})
	}
}
