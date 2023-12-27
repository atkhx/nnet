package num

import (
	"context"

	"github.com/atkhx/mps"
)

func NewData(data, grad *mps.MTLBuffer, dims Dims, srcNodes ...*Data) *Data {
	return &Data{
		Data:     data,
		Grad:     grad,
		Dims:     dims,
		SrcNodes: srcNodes,
	}
}

type Data struct {
	Data *mps.MTLBuffer
	Grad *mps.MTLBuffer

	Dims Dims

	SrcNodes Nodes
	CalcData func(ctx context.Context)
	CalcGrad func(ctx context.Context)

	SkipResetGrad bool
}
