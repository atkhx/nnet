package conv

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestFilterSize(t *testing.T) {
	layer := &layer{}

	FilterSize(15)(layer)

	assert.Equal(t, layer.FWidth, 15)
	assert.Equal(t, layer.FHeight, 15)
}

func TestFiltersCount(t *testing.T) {
	layer := &layer{}

	FiltersCount(17)(layer)
	assert.Equal(t, layer.FCount, 17)
}

func TestPadding(t *testing.T) {
	layer := &layer{}

	Padding(3)(layer)
	assert.Equal(t, layer.FPadding, 3)
}

func TestStride(t *testing.T) {
	layer := &layer{}

	Stride(7)(layer)
	assert.Equal(t, layer.FStride, 7)
}
