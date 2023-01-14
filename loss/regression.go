package loss

import (
	"math"

	"github.com/atkhx/nnet/data"
)

func NewRegression() *Regression {
	return &Regression{}
}

type Regression struct{}

//nolint:gomnd
func (c *Regression) GetError(target, result []float64) (res float64) {
	for i := 0; i < len(target); i++ {
		res += math.Pow(result[i]-target[i], 2)
	}
	return 0.5 * res
}

func (c *Regression) GetDeltas(target, output *data.Data) (deltas *data.Data) {
	deltas = output.Copy()
	for i, v := range target.Data {
		deltas.Data[i] -= v
	}
	return
}
