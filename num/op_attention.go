package num

import "math"

type SAHeadWeights struct {
	KeyWeights *Data
	QryWeights *Data
	ValWeights *Data
}

func NewSAHeadWeights(headSize, featuresCount int, weightK float64) SAHeadWeights {
	return SAHeadWeights{
		KeyWeights: NewRandNormWeighted(NewDims(headSize, featuresCount, 1), weightK),
		QryWeights: NewRandNormWeighted(NewDims(headSize, featuresCount, 1), weightK),
		ValWeights: NewRandNormWeighted(NewDims(headSize, featuresCount, 1), weightK),
	}
}

// https://www.youtube.com/playlist?list=PLDw5cZwIToCvXLVY2bSqt7F2gu8y-Rqje

func (aData *Data) SAMasked(
	dropoutProb float64,
	headWeights SAHeadWeights,
) *Data {
	// k = 1 / sqrt(headSize)
	k := math.Pow(float64(headWeights.QryWeights.Dims.W), -0.5)

	// This is the mask attention https://youtu.be/XowwKOAWYoQ?t=1664
	keyObject := aData.MatrixMultiply(headWeights.KeyWeights)
	qryObject := aData.MatrixMultiply(headWeights.QryWeights)
	valObject := aData.MatrixMultiply(headWeights.ValWeights)

	weiObject := keyObject.MatrixMultiply(qryObject.Transpose(), WithMatrixMultiplyAlpha(k))
	weiSoftmax := weiObject.TriangleLowerSoftmax()

	if dropoutProb > 0 && dropoutProb < 1 {
		weiSoftmax = weiSoftmax.Dropout(dropoutProb)
	}

	return weiSoftmax.MatrixMultiply(valObject)
}

func (aData *Data) SA(
	dropoutProb float64,
	headWeights SAHeadWeights,
) *Data {
	// k = 1 / sqrt(headSize)
	k := math.Pow(float64(headWeights.QryWeights.Dims.W), -0.5)

	// This is the mask attention https://youtu.be/XowwKOAWYoQ?t=1664
	keyObject := aData.MatrixMultiply(headWeights.KeyWeights)
	qryObject := aData.MatrixMultiply(headWeights.QryWeights)
	valObject := aData.MatrixMultiply(headWeights.ValWeights)

	weiObject := keyObject.MatrixMultiply(qryObject.Transpose(), WithMatrixMultiplyAlpha(k))
	weiSoftmax := weiObject.Softmax()

	if dropoutProb > 0 && dropoutProb < 1 {
		weiSoftmax = weiSoftmax.Dropout(dropoutProb)
	}

	return weiSoftmax.MatrixMultiply(valObject)
}
