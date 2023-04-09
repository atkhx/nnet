package fc

import (
	"fmt"
	"testing"

	"github.com/atkhx/nnet/data"
)

func TestFC_AsEmbedding(t *testing.T) {
	featuresCount := 2
	embed := New(
		WithInputSize(5),
		WithLayerSize(featuresCount),
	)

	wordLength := 2
	input := data.NewOneHotVectors(5,
		1, 0,
		1, 3,
		2, 1,
		0, 2,
	)

	fmt.Println("embed.Weights.Data", embed.Weights.Data)

	output := embed.Forward(input)
	output = data.WrapData(
		wordLength*featuresCount,
		output.Data.GetLen()/(wordLength*featuresCount),
		1,
		output.Data.Data,
	)

	fmt.Println("input", input.Data)
	fmt.Println("output", output.Data)

}
