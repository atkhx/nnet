package data

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestConvLayerTo2(t *testing.T) {
	iw, ih := 5, 5
	fw, fh := 3, 3
	// todo fix for padding 2
	padding := 1

	ow, oh := CalcConvOutputSize(iw, ih, fw, fh, padding, 1)

	fmt.Println("ow", ow, "oh", oh)
	fmt.Println("iw", iw, "ih", ih)
	fmt.Println()

	inputs := make([]float64, iw*ih)
	FillRandom(inputs)

	iwP, ihP, _, inputsP := AddPadding(inputs, iw, ih, 1, padding)

	filter := make([]float64, fw*fh)
	FillRandom(filter)

	output1 := make([]float64, ow*oh)
	output4 := make([]float64, ow*oh)

	ConvLayerTo(ow, oh, output1, iwP, ihP, inputsP, fw, fh, filter)
	ConvTo(ow, oh, output4, iw, ih, inputs, fw, fh, filter, 1, padding)

	assert.True(t, true)
	assert.Equal(t, output1, output4)

	fmt.Println(output4)

	// for oy := oyOffset; oy < oh-(oh-ih-oyOffset); oy++ {

}
