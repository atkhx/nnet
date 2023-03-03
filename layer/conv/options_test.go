package conv

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestConvFilterSize(t *testing.T) {
	layer := &Conv{}

	FilterSize(15)(layer)

	assert.Equal(t, layer.FWidth, 15)
	assert.Equal(t, layer.FHeight, 15)
}

func TestConvFiltersCount(t *testing.T) {
	layer := &Conv{}

	FiltersCount(17)(layer)
	assert.Equal(t, layer.FCount, 17)
}

func TestConvPadding(t *testing.T) {
	layer := &Conv{}

	Padding(3)(layer)
	assert.Equal(t, layer.FPadding, 3)
}

func TestConvStride(t *testing.T) {
	layer := &Conv{}

	Stride(7)(layer)
	assert.Equal(t, layer.FStride, 7)
}
