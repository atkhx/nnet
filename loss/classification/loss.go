package classification

import (
	"github.com/atkhx/nnet/data"
	"math"
)

const minimalNonZeroFloat = 0.000000000000000000001

func New() *loss {
	return &loss{}
}

type loss struct{}

func (c *loss) GetError(target, output []float64) float64 {
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

func (c *loss) GetDeltas(target, output *data.Data) (res *data.Data) {
	res = output.Copy()
	for i := 0; i < len(target.Data); i++ {
		res.Data[i] = output.Data[i] - target.Data[i]
	}
	return
}
