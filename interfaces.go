package nnet

import (
	"context"

	"github.com/atkhx/nnet/num"
)

type Device interface {
	Release()

	NewData(dims num.Dims, srcNodes ...*num.Data) *num.Data
	NewDataWithValues(dims num.Dims, values []float32) *num.Data

	NewDataRandNormWeighted(dims num.Dims, w float32) *num.Data

	NewTokenEmbeddingTable(featuresCount, alphabetSize int) *num.Data
	NewPositionEmbeddingTable(featuresCount, contextSize int) *num.Data

	GetDataDims(aData *num.Data) num.Dims
	GetDataLength(aData *num.Data) int
	RMSNorm(aData *num.Data, width int) *num.Data
	Relu(aData *num.Data) *num.Data
	SiLu(input *num.Data) *num.Data
	Rope(input *num.Data, headIndex, headSize, contextLength int) *num.Data
	RopeCols(input *num.Data, featuresCount, headSize, contextLength int) *num.Data
	RopeRows(input *num.Data, featuresCount, headSize, contextLength int) *num.Data

	Add(aData *num.Data, bData *num.Data) *num.Data
	AddRow(input, weights *num.Data, width int) *num.Data
	AddEqual(input, weights *num.Data) *num.Data
	MulRow(inputs, weights *num.Data, width int) *num.Data
	MulEqual(input, weights *num.Data) *num.Data
	ConcatByRows(bData ...*num.Data) *num.Data
	Dropout(aData *num.Data, prob float32) *num.Data
	Reshape(aData *num.Data, dims num.Dims) *num.Data
	Embeddings(aData *num.Data, tEmbeddings *num.Data) *num.Data
	Transpose(aData *num.Data) *num.Data
	TriangleLowerSoftmax(aData *num.Data) *num.Data
	MatrixMultiply(aData *num.Data, bData *num.Data, alpha float32) *num.Data

	GetOptimizerAdam(iterations int, beta1, beta2, learningRate, eps float32) func(nodes []*num.Data) func(ctx context.Context, iteration int)
}

type Layer interface {
	Compile(device Device, inputs *num.Data) *num.Data
}

type LayerUpdatable interface {
	ForUpdate() []*num.Data
}

type LayerWithWeightsProvider interface {
	LoadFromProvider()
}
