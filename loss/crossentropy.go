package loss

import (
	"math"

	"github.com/atkhx/nnet/num"
)

func CrossEntropy(target, actual num.Float64s, bSize int) (loss float64) {
	chunkSize := len(actual) / bSize

	actual = actual.Copy()
	for i := 0; i < len(actual); i += chunkSize {
		actual[i : i+chunkSize].Softmax()
	}

	for i, t := range target {
		actual[i] = -t * math.Log(actual[i])
	}

	return actual.Sum() / float64(bSize)
}

func CrossEntropyBackward(target, actual num.Float64s, bSize int) (oGrads num.Float64s) {
	chunkSize := len(actual) / bSize

	oGrads = actual.Copy()
	for i := 0; i < len(actual); i += chunkSize {
		oGrads[i : i+chunkSize].Softmax()
	}

	k := 1.0 / float64(bSize)
	for i, t := range target {
		oGrads[i] = k * (oGrads[i] - t)
	}
	return oGrads
}
