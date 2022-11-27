package fc

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestOutputSizes(t *testing.T) {
	layer := &FC{}

	OutputSizes(3, 4, 5)(layer)

	assert.Equal(t, layer.OWidth, 3)
	assert.Equal(t, layer.OHeight, 4)
	assert.Equal(t, layer.ODepth, 5)
}
