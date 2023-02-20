package loss

import "github.com/atkhx/nnet/data"

type GetLossFunc func(output *data.Data) LossObject

type LossObject interface {
	GetError() float64
	GetGradient() *data.Data
}
