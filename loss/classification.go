package loss

import (
	"math"

	"github.com/atkhx/nnet/data"
)

const minimalNonZeroFloat = 0.000000000000000000001

func NewClassification() *Classification {
	return &Classification{}
}

type Classification struct{}

func (c *Classification) GetError(target, output []float64) float64 {
	for i := 0; i < len(target); i++ {
		if target[i] == 1 {
			if output[i] == 0 {
				return -math.Log(minimalNonZeroFloat)
			} else {
				return -math.Log(output[i])
			}
		}
	}
	return 0
}

// GetDeltas - honest implementation of cross-entropy derivative.
// Works with honest implementation of softmax layer.
// https://deepnotes.io/softmax-crossentropy
func (c *Classification) GetDeltas(target, output *data.Data) (deltas *data.Data) {
	deltas = output.CopyZero()
	for i, t := range target.Data {
		o := output.Data[i]
		deltas.Data[i] = -(t / o) + ((1 - t) / (1 - o))
	}
	return
}

// GetDeltasCheat - quick implementation of derivative cross-entropy + softmax.
// Works only with softmax.BackwardCheat implementation.
// todo use quick functions by options in layers or special block.
func (c *Classification) GetDeltasCheat(target, output *data.Data) (deltas *data.Data) {
	deltas = output.Copy()
	for i, v := range target.Data {
		deltas.Data[i] -= v
	}
	return
}
