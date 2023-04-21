package layer

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/atkhx/nnet/num"
)

func TestEmbedPos_Forward(t *testing.T) {
	alphabetSize := 5
	featuresCount := 3
	contextLength := 2
	batchSize := 1

	emb := NewEmbedPos(featuresCount, alphabetSize, 0.0)
	inputs := make(num.Float64s, contextLength*batchSize)
	iGrads := make(num.Float64s, contextLength*batchSize)

	output, oGrads := emb.Compile(batchSize, inputs, iGrads)

	require.Len(t, emb.Weights, alphabetSize*featuresCount)
	require.Len(t, oGrads, featuresCount*contextLength*batchSize)

	copy(emb.Weights, num.Float64s{
		1, 2, 3,
		3, 4, 5,
		4, 5, 6,
		5, 6, 7,
		6, 7, 8,
	})

	copy(inputs, num.Float64s{
		0, 1,
	})

	for i := 0; i < len(emb.Weights); i += alphabetSize {
		fmt.Println(emb.Weights[i : i+alphabetSize])
	}

	emb.Forward()

	//require.Equal(t, num.Float64s{1, 2, 3, 7, 8, 9}, output)

	fmt.Println("output", output)
	fmt.Println("oGrads", oGrads)

	//copy(oGrads, num.Float64s{0.1, 0.1, 0.1, 0.1, 0.1, 0.1})

	emb.Backward()
}
