package num

import "math"

func (input *Data) SAHead(
	headSize int,

	keyWeights *Data,
	qryWeights *Data,
	valWeights *Data,
) *Data {
	k := math.Pow(float64(headSize), -0.5)

	keyObject := input.MatrixMultiply(keyWeights)
	qryObject := input.MatrixMultiply(qryWeights)
	valObject := input.MatrixMultiply(valWeights)

	qryObjectT := qryObject.Transpose()
	weiObject := keyObject.MatrixMultiply(qryObjectT)
	weiMulObject := weiObject.MulScalar(k)

	weiTrilObject := weiMulObject.TriangleLower(math.Inf(-1))
	weiSoftmaxObject := weiTrilObject.Softmax()

	output := weiSoftmaxObject.MatrixMultiply(valObject)
	outputCalcDataLast := output.calcData

	output.calcData = func() {
		keyObject.Forward()
		qryObject.Forward()
		valObject.Forward()

		qryObjectT.Forward()
		weiObject.Forward()
		weiMulObject.Forward()
		weiTrilObject.Forward()
		weiSoftmaxObject.Forward()

		outputCalcDataLast()
	}

	return output
}
