package conv

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestConvFilterSize(t *testing.T) {
	layer := &Conv{}

	WithFilterSize(15)(layer)

	assert.Equal(t, layer.FilterSize, 15)
	assert.Equal(t, layer.FilterSize, 15)
}

func TestConvFiltersCount(t *testing.T) {
	layer := &Conv{}

	WithFiltersCount(17)(layer)
	assert.Equal(t, layer.FiltersCount, 17)
}

func TestConvPadding(t *testing.T) {
	layer := &Conv{}

	WithPadding(3)(layer)
	assert.Equal(t, layer.Padding, 3)
}

func TestConvStride(t *testing.T) {
	layer := &Conv{}

	WithStride(7)(layer)
	assert.Equal(t, layer.Stride, 7)
}
