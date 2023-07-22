package num

// https://www.youtube.com/playlist?list=PLDw5cZwIToCvXLVY2bSqt7F2gu8y-Rqje

func (aData *Data) MaskedSelfAttention(
	k float64,
	keyWeights *Data,
	qryWeights *Data,
	valWeights *Data,
) *Data {
	// This is the mask attention https://youtu.be/XowwKOAWYoQ?t=1664
	keyObject := aData.MatrixMultiply(keyWeights)
	qryObject := aData.MatrixMultiply(qryWeights)
	valObject := aData.MatrixMultiply(valWeights)

	weiObject := keyObject.MatrixMultiply(qryObject.Transpose(), WithMatrixMultiplyAlpha(k))
	weiSoftmax := weiObject.TriangleLowerSoftmax()

	return weiSoftmax.MatrixMultiply(valObject)
}
