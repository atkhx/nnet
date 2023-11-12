package nnet

import (
	"github.com/atkhx/nnet/num"
)

type Device interface {
	Close() error

	NewData(dims num.Dims, srcNodes ...*num.Data) *num.Data
	NewDataRandNormWeighted(dims num.Dims, w float32) *num.Data

	NewTokenEmbeddingTable(featuresCount, alphabetSize int) *num.Data
	NewPositionEmbeddingTable(featuresCount, contextSize int) *num.Data

	//FillDataWithZeros(aData *num.Data)
	//FillDataWithOnes(aData *num.Data)

	GetDataDims(aData *num.Data) num.Dims
	GetDataLength(aData *num.Data) int

	//Sqrt(aData *num.Data) *num.Data
	//Mean(aData *num.Data) *num.Data
	//MeanByRows(aData *num.Data) *num.Data
	//VarianceByRows(aData *num.Data, mean *num.Data) *num.Data

	//LNorm(aData, gamma, beta *num.Data) *num.Data
	RMSNorm(aData *num.Data) *num.Data

	Relu(aData *num.Data) *num.Data

	//AddScalar(aData *num.Data, k float32) *num.Data
	//MulScalar(aData *num.Data, k float32) *num.Data

	Add(aData *num.Data, bData *num.Data) *num.Data
	//Sub(aData *num.Data, bData *num.Data) *num.Data
	//Mul(aData *num.Data, bData *num.Data) *num.Data
	//Div(aData *num.Data, bData *num.Data) *num.Data

	ConcatByRows(bData ...*num.Data) *num.Data
	Dropout(aData *num.Data, prob float32) *num.Data
	Reshape(aData *num.Data, dims num.Dims) *num.Data
	//CrossEntropyPos(aData *num.Data, targets *num.Data) *num.Data
	Embeddings(aData *num.Data, tEmbeddings, pEmbeddings *num.Data) *num.Data
	Transpose(aData *num.Data) *num.Data
	TriangleLowerSoftmax(aData *num.Data) *num.Data
	//MatrixMultiply2D(aData *num.Data, bData *num.Data, alpha float32) *num.Data
	MatrixMultiply3D(aData *num.Data, bData *num.Data, alpha float32) *num.Data

	GetOptimizerAdam(iterations int, beta1, beta2, learningRate, eps float32) func(nodes []*num.Data) func(iteration int)
}

type Layer interface {
	Compile(device Device, inputs *num.Data) *num.Data
}

type LayerUpdatable interface {
	ForUpdate() []*num.Data
}
