package loss

import (
	"math"

	"github.com/atkhx/nnet/data"
)

const minimalNonZeroFloat = 0.000000000000000000001

func NewClassificationLossFunc(target *data.Data) GetLossFunc {
	return func(output *data.Data) LossObject {
		return &Classification{
			Target: target,
			Output: output,
		}
	}
}

type Classification struct {
	Target *data.Data
	Output *data.Data
}

func (c *Classification) GetError() float64 {
	for i := 0; i < len(c.Target.Data); i++ {
		if c.Target.Data[i] == 1 {
			if c.Output.Data[i] == 0 {
				return -math.Log(minimalNonZeroFloat)
			} else {
				return -math.Log(c.Output.Data[i])
			}
		}
	}
	return 0
}

func (c *Classification) GetGradient() (lossGradient *data.Data) {
	lossGradient = c.Output.CopyZero()
	for i, t := range c.Target.Data {
		o := c.Output.Data[i]
		lossGradient.Data[i] = -(t / o) + ((1 - t) / (1 - o))
	}
	return
}
