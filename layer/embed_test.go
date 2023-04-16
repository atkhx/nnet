package layer

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/atkhx/nnet/num"
)

func TestEmbed_Forward(t *testing.T) {
	alphabetSize := 5
	featuresCount := 3
	contextLength := 2
	batchSize := 1

	emb := NewEmbed(featuresCount, contextLength, 0.0)
	inputs := make(num.Float64s, alphabetSize*contextLength*batchSize)
	iGrads := make(num.Float64s, alphabetSize*contextLength*batchSize)

	output, oGrads := emb.Compile(batchSize, inputs, iGrads)

	require.Len(t, emb.Weights, alphabetSize*featuresCount)
	require.Len(t, oGrads, featuresCount*contextLength*batchSize)
	require.Len(t, oGrads, featuresCount*contextLength*batchSize)

	copy(emb.Weights, num.Float64s{
		1, 2, 3, 4, 7,
		2, 3, 4, 5, 8,
		3, 4, 5, 6, 9,
	})

	copy(inputs, num.Float64s{
		1, 0, 0, 0, 0,
		0, 0, 0, 0, 1,
	})

	for i := 0; i < len(emb.Weights); i += alphabetSize {
		fmt.Println(emb.Weights[i : i+alphabetSize])
	}

	emb.Forward()

	require.Equal(t, num.Float64s{1, 2, 3, 7, 8, 9}, output)

	fmt.Println("output", output)
	fmt.Println("oGrads", oGrads)

	copy(oGrads, num.Float64s{0.1, 0.1, 0.1, 0.1, 0.1, 0.1})

	emb.Backward()
}
