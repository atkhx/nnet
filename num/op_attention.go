package num

import (
	"math"
)

// https://www.youtube.com/playlist?list=PLDw5cZwIToCvXLVY2bSqt7F2gu8y-Rqje

func (aData *Data) MaskedSelfAttention(
	headSize int,

	keyWeights *Data,
	qryWeights *Data,
	valWeights *Data,
) *Data {
	// This is the mask attention https://youtu.be/XowwKOAWYoQ?t=1664
	k := math.Pow(float64(headSize), -0.5)

	keyObject := aData.MatrixMultiply(keyWeights)
	qryObject := aData.MatrixMultiply(qryWeights)
	valObject := aData.MatrixMultiply(valWeights)

	weiObject := keyObject.MatrixMultiply(qryObject.Transpose())
	weiSoftmax := weiObject.TriangleLowerSoftmax(k)

	return weiSoftmax.MatrixMultiply(valObject)
	//return weiSoftmax.TriangleLowerMatrixMultiply(valObject)
}

func (aData *Data) MaskedSelfAttention2(
	headSize int,

	keyWeights *Data,
	qryWeights *Data,
	valWeights *Data,
) *Data {
	// This is the mask attention https://youtu.be/XowwKOAWYoQ?t=1664
	k := math.Pow(float64(headSize), -0.5)

	keyObject := aData.MatrixMultiply(keyWeights)
	qryObject := aData.MatrixMultiply(qryWeights)
	valObject := aData.MatrixMultiply(valWeights)

	weiObject := keyObject.MatrixMultiply(qryObject.Transpose())
	weiSoftmax := weiObject.TriangleLowerSoftmax(k)

	return weiSoftmax.TriangleLowerMatrixMultiply(valObject)
}

func (aData *Data) MaskedCrossAttention(
	bData *Data, // data from the encoder

	headSize int,

	keyWeights *Data,
	qryWeights *Data,
	valWeights *Data,
) *Data {
	// This is the cross attention https://youtu.be/XowwKOAWYoQ?t=1834
	// bData - output from the encoder
	// aData - output from the positional embedding table of the decoder

	k := math.Pow(float64(headSize), -0.5)

	keyObject := bData.MatrixMultiply(keyWeights) // from the encoder
	qryObject := aData.MatrixMultiply(qryWeights)
	valObject := bData.MatrixMultiply(valWeights) // from the encoder

	weiObject := keyObject.MatrixMultiply(qryObject.Transpose())
	weiSoftmax := weiObject.TriangleLowerSoftmax(k)

	return weiSoftmax.TriangleLowerMatrixMultiply(valObject)
}

func (aData *Data) SelfAttention(
	headSize int,

	keyWeights *Data,
	qryWeights *Data,
	valWeights *Data,
) *Data {
	k := math.Pow(float64(headSize), -0.5)

	keyObject := aData.MatrixMultiply(keyWeights)
	qryObject := aData.MatrixMultiply(qryWeights)
	valObject := aData.MatrixMultiply(valWeights)

	weiObject := keyObject.MatrixMultiply(qryObject.Transpose())
	weiSoftmax := weiObject.MulScalar(k)

	return weiSoftmax.MatrixMultiply(valObject)
}
