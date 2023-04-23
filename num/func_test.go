package num

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_getRepeatedPosPairs(t *testing.T) {
	type testCase struct {
		aLen int
		bLen int

		expected [][2]int
	}

	testCases := map[string]testCase{
		"empty": {
			aLen:     0,
			bLen:     0,
			expected: [][2]int{},
		},
		"first is scalar": {
			aLen: 1,
			bLen: 3,
			expected: [][2]int{
				{0, 0},
				{0, 1},
				{0, 2},
			},
		},
		"second is scalar": {
			aLen: 3,
			bLen: 1,
			expected: [][2]int{
				{0, 0},
				{1, 0},
				{2, 0},
			},
		},
		"both are scalar": {
			aLen: 1,
			bLen: 1,
			expected: [][2]int{
				{0, 0},
			},
		},
		"slice A is shorter than slice B": {
			aLen: 2,
			bLen: 4,
			expected: [][2]int{
				{0, 0},
				{1, 1},
				{0, 2},
				{1, 3},
			},
		},
		"slice A is longer than slice B": {
			aLen: 6,
			bLen: 2,
			expected: [][2]int{
				{0, 0},
				{1, 1},
				{2, 0},
				{3, 1},
				{4, 0},
				{5, 1},
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			require.Equal(t, tc.expected, GetRepeatedPosPairs(tc.aLen, tc.bLen))
		})
	}
}

func Test_MaxOnInt(t *testing.T) {
	type testCase struct {
		a, b, expected int
	}

	testCases := map[string]testCase{
		"A and B are equals": {7, 7, 7},
		"A less than B":      {3, 8, 8},
		"B less than A":      {4, -1, 4},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			require.Equal(t, tc.expected, max(tc.a, tc.b))
		})
	}
}

func Test_MinOnInt(t *testing.T) {
	type testCase struct {
		a, b, expected int
	}

	testCases := map[string]testCase{
		"A and B are equals": {7, 7, 7},
		"A less than B":      {3, 8, 3},
		"B less than A":      {4, -1, -1},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			require.Equal(t, tc.expected, min(tc.a, tc.b))
		})
	}
}
