package nnet

import (
	"github.com/atkhx/nnet/num"
)

type Device[data any] interface {
	NewData(dims num.Dims, srcNodes ...data) data
	NewDataRandNormWeighted(dims num.Dims, w float32) data

	NewTokenEmbeddingTable(featuresCount, alphabetSize int) data
	NewPositionEmbeddingTable(featuresCount, contextSize int) data

	FillDataWithZeros(aData data)
	FillDataWithOnes(aData data)

	GetDataDims(aData data) num.Dims
	GetDataLength(aData data) int

	Sqrt(aData data) data
	Mean(aData data) data
	MeanByRows(aData data) data
	VarianceByRows(aData data, mean data) data

	LNorm(aData, gamma, beta data) data

	Relu(aData data) data

	AddScalar(aData data, k float32) data
	MulScalar(aData data, k float32) data

	Add(aData data, bData data) data
	Sub(aData data, bData data) data
	Mul(aData data, bData data) data
	Div(aData data, bData data) data

	ConcatByRows(bData ...data) data
	Dropout(aData data, prob float32) data
	Reshape(aData data, dims num.Dims) data
	CrossEntropyPos(aData data, targets data) data
	Embeddings(aData data, tEmbeddings, pEmbeddings data) data
	Transpose(aData data) data
	TriangleLowerSoftmax(aData data) data
	MatrixMultiply2D(aData data, bData data, options ...num.MMOption) data
	MatrixMultiply(aData data, bData data, options ...num.MMOption) data
}

type Layer[data any] interface {
	Compile(device Device[data], inputs data) data
}

type LayerUpdatable[data any] interface {
	ForUpdate() []data
}
