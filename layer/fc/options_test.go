package fc

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestOutputSizes(t *testing.T) {
	layer := &layer{}

	OutputSizes(3, 4, 5)(layer)

	assert.Equal(t, layer.OWidth, 3)
	assert.Equal(t, layer.OHeight, 4)
	assert.Equal(t, layer.ODepth, 5)
}
