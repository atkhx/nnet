package num

import (
	"context"
)

type Nodes []*Data

func NewData(data, grad []float32, dims Dims, opts any, srcNodes ...*Data) *Data {
	return &Data{
		Data:     data,
		Grad:     grad,
		Dims:     dims,
		Opts:     opts,
		SrcNodes: srcNodes,
	}
}

type Data struct {
	Data []float32
	Grad []float32 `json:"-"`
	Dims Dims
	Opts any `json:"-"`

	SrcNodes Nodes                     `json:"-"`
	CalcData func(ctx context.Context) `json:"-"`
	CalcGrad func(ctx context.Context) `json:"-"`

	SkipResetGrad bool `json:"-"`
}
