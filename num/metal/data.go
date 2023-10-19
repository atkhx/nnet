package metal

import (
	"context"
)

type Nodes []*Data
type Data struct {
	Data Float32s
	Grad Float32s `json:"-"`
	Dims Dims

	srcNodes      Nodes
	calcData      func(ctx context.Context)
	calcGrad      func(ctx context.Context)
	skipResetGrad bool

	//dataBuffer *mps.MTLBuffer
	//gradBuffer *mps.MTLBuffer
}

func (aData *Data) SkipResetGrad() {
	aData.skipResetGrad = true
}

func New(dims Dims, srcNodes ...*Data) *Data {
	//dataBuffer := mps.DefaultDevice.CreateNewBufferWithLength(dims.Size())
	//gradBuffer := mps.DefaultDevice.CreateNewBufferWithLength(dims.Size())

	return &Data{
		//Data: dataBuffer.GetData(),
		//Grad: gradBuffer.GetData(),
		Data: NewFloat32s(dims.Size()),
		Grad: NewFloat32s(dims.Size()),
		Dims: dims,

		srcNodes: srcNodes,

		//dataBuffer: dataBuffer,
		//gradBuffer: gradBuffer,
	}
}

func NewRandNorm(dims Dims) *Data {
	data := New(dims)
	data.Data.CopyFrom(NewRandNormFloat32s(dims.Size()))
	return data
}

func NewRandNormWeighted(dims Dims, w float32) *Data {
	data := New(dims)
	data.Data.CopyFrom(NewRandNormWeightedFloat32s(dims.Size(), w))
	return data
}

func (aData *Data) NewLinkedCopy() *Data {
	return New(aData.Dims, aData)
}

func (aData *Data) StringData() string {
	return aData.Data.String(aData.Dims)
}

func (aData *Data) StringGrad() string {
	return aData.Grad.String(aData.Dims)
}

func (aData *Data) Forward(ctx context.Context) {
	aData.calcData(ctx)
}
