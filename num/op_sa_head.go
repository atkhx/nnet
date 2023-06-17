package num

import (
	"math"
)

func (aData *Data) SAHead(
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
	weiSoftmax := weiObject.TriangleLowerSoftmax(k)

	return weiSoftmax.TriangleLowerMatrixMultiply(valObject)
}

func (aData *Data) SAHeadTransposed(
	headSize int,

	keyWeights *Data,
	qryWeights *Data,
	valWeights *Data,
) *Data {
	k := math.Pow(float64(headSize), -0.5)

	keyObject := aData.MatrixMultiplyTransposed(keyWeights)
	qryObject := aData.MatrixMultiplyTransposed(qryWeights)
	valObject := aData.MatrixMultiplyTransposed(valWeights)

	weiObject := keyObject.MatrixMultiplyTransposed(qryObject)
	weiSoftmax := weiObject.TriangleLowerSoftmax(k)

	return weiSoftmax.TriangleLowerMatrixMultiply(valObject)
}
