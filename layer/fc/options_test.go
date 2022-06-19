package fc

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestOutputSizes(t *testing.T) {
	layer := &layer{}

	OutputSizes(3, 4, 5)(layer)

	assert.Equal(t, layer.oWidth, 3)
	assert.Equal(t, layer.oHeight, 4)
	assert.Equal(t, layer.oDepth, 5)
}
