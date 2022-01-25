package regression

import (
	"github.com/atkhx/nnet/data"
	"math"
)

func New() *loss {
	return &loss{}
}

type loss struct{}

func (c *loss) GetError(target, result []float64) (res float64) {
	for i := 0; i < len(target); i++ {
		res += math.Pow(result[i]-target[i], 2)
	}
	return 0.5 * res
}

func (c *loss) GetDeltas(target, output *data.Data) (res *data.Data) {
	res = output.CopyZero()
	for i := 0; i < len(target.Data); i++ {
		res.Data[i] = output.Data[i] - target.Data[i]
	}
	return
}
