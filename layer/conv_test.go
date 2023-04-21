package layer

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/atkhx/nnet/num"
)

func TestConv_Forward(t *testing.T) {
	imageSize := 10
	imageDepth := 3

	filterSize := 3
	filtersCount := 3

	batchSize := 3
	padding := 0

	inputs := make(num.Float64s, imageSize*imageSize*imageDepth*batchSize)
	iGrads := make(num.Float64s, imageSize*imageSize*imageDepth*batchSize)

	conv := NewConv(imageSize, imageDepth, filterSize, filtersCount, padding, num.ReLuGain)
	conv.Compile(3, inputs, iGrads)

	require.Equal(t, filterSize*filterSize*imageDepth*filtersCount, len(conv.Weights))
	require.Equal(t, filterSize*filterSize*imageDepth*filtersCount, len(conv.wGrads))

	require.Equal(t, 8*8*filtersCount*batchSize, len(conv.output))
	require.Equal(t, 8*8*filtersCount*batchSize, len(conv.oGrads))
}
