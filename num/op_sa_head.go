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

	keyWeights.Name += "keyWeights"
	qryWeights.Name += "qryWeights"
	valWeights.Name += "valWeights"

	keyObject := aData.MatrixMultiply(keyWeights)
	qryObject := aData.MatrixMultiply(qryWeights)
	valObject := aData.MatrixMultiply(valWeights)

	weiObject := keyObject.MatrixMultiply(qryObject.Transpose())
	weiSoftmax := weiObject.TriangleLowerSoftmax(k)

	return weiSoftmax.TriangleLowerMatrixMultiply(valObject)
}
