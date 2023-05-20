package num

import (
	"math"
	"sync"
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

	qryObjectT := qryObject.Transpose()
	weiObject := keyObject.MatrixMultiply(qryObjectT)
	weiMulObject := weiObject.MulScalar(k)

	weiTrilObject := weiMulObject.TriangleLower(math.Inf(-1))
	weiSoftmaxObject := weiTrilObject.Softmax()

	valObject := aData.MatrixMultiply(valWeights)

	output := weiSoftmaxObject.MatrixMultiply(valObject)
	weiSoftmaxObjectMulValObjectForward := output.calcData

	wg := sync.WaitGroup{}
	output.calcData = func() {
		wg.Add(2)
		go func() {
			keyObject.Forward()
			wg.Done()
		}()

		go func() {
			qryObject.Forward()
			wg.Done()
		}()

		wg.Wait()

		wg.Add(2)
		go func() {
			valObject.Forward()
			wg.Done()
		}()

		go func() {
			qryObjectT.Forward()
			weiObject.Forward()
			weiMulObject.Forward()
			weiTrilObject.Forward()
			weiSoftmaxObject.Forward()
			wg.Done()
		}()

		wg.Wait()
		weiSoftmaxObjectMulValObjectForward()
	}

	weiSoftmaxObjectMulValObjectBackward := output.calcGrad
	output.calcGrad = func() {
		weiSoftmaxObjectMulValObjectBackward()
		weiSoftmaxObject.Backward()

		weiTrilObject.Backward()
		weiMulObject.Backward()
		weiObject.Backward()
		qryObjectT.Backward()

		valObject.Backward()
		qryObject.Backward()
		keyObject.Backward()
	}

	return output
}
