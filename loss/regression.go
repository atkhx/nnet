package loss

import (
	"math"

	"github.com/atkhx/nnet/data"
)

func NewRegressionLossFunc(target *data.Data) GetLossFunc {
	return func(output *data.Data) LossObject {
		return &Regression{
			Target: target,
			Output: output,
		}
	}
}

type Regression struct {
	Target *data.Data
	Output *data.Data
}

//nolint:gomnd
func (c *Regression) GetError() (res float64) {
	for i := 0; i < len(c.Target.Data); i++ {
		res += math.Pow(c.Output.Data[i]-c.Target.Data[i], 2)
	}
	return 0.5 * res
}

func (c *Regression) GetGradient() (lossGradient *data.Data) {
	// todo validate algorithm
	lossGradient = c.Output.Copy()
	for i, v := range c.Target.Data {
		lossGradient.Data[i] -= v
	}
	return
}
