package native

import (
	"context"

	"github.com/atkhx/nnet/num"
)

type Nodes []*Data

type Data struct {
	Data Float32s
	Grad Float32s `json:"-"`
	Dims num.Dims

	srcNodes      Nodes
	calcData      func(ctx context.Context)
	calcGrad      func(ctx context.Context)
	skipResetGrad bool
}

func (aData *Data) SkipResetGrad() {
	aData.skipResetGrad = true
}

func (aData *Data) Forward(ctx context.Context) {
	aData.calcData(ctx)
}
