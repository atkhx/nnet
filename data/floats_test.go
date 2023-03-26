package data

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestConvLayerTo2(t *testing.T) {
	iw, ih := 32, 32
	fw, fh := 3, 3
	padding := 2

	ow, oh := CalcConvOutputSize(iw, ih, fw, fh, padding, 1)

	inputs := make([]float64, iw*ih)
	FillRandom(inputs)

	iw, ih, _, inputs = AddPadding(inputs, iw, ih, 1, padding)

	filter := make([]float64, fw*fh)
	FillRandom(filter)

	output1 := make([]float64, ow*oh)
	output3 := make([]float64, ow*oh)

	ConvLayerTo(ow, oh, output1, iw, ih, inputs, fw, fh, filter)
	ConvolveTo(ow, oh, output3, iw, ih, inputs, fw, fh, filter, 1, 0)

	assert.Equal(t, output1, output3)
}
