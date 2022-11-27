package conv

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFilterSize(t *testing.T) {
	layer := &Layer{}

	FilterSize(15)(layer)

	assert.Equal(t, layer.FWidth, 15)
	assert.Equal(t, layer.FHeight, 15)
}

func TestFiltersCount(t *testing.T) {
	layer := &Layer{}

	FiltersCount(17)(layer)
	assert.Equal(t, layer.FCount, 17)
}

func TestPadding(t *testing.T) {
	layer := &Layer{}

	Padding(3)(layer)
	assert.Equal(t, layer.FPadding, 3)
}

func TestStride(t *testing.T) {
	layer := &Layer{}

	Stride(7)(layer)
	assert.Equal(t, layer.FStride, 7)
}
