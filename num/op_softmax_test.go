package num

import (
	"fmt"
	"math"
	"testing"
)

func TestData_TriangleLowerSoftmax(t *testing.T) {
	inputs := NewRandNormWeighted(NewDims(5, 5), 0.02)
	trilsm := inputs.TriangleLowerSoftmax(0)
	mean := trilsm.Mean()

	trilsm.Forward()
	mean.Forward()

	mean.ResetGrads(0.00001)
	mean.Backward()

	trilsm.Backward()

	fmt.Println(trilsm.StringData())
	fmt.Println(trilsm.StringGrad())
	fmt.Println(inputs.StringGrad())
}

func TestData_TriangleLowerSoftmax2(t *testing.T) {
	inputs := NewRandNormWeighted(NewDims(5, 5), 0.02)
	//inputs.Data = Float64s{
	//	1, 2, 3, 4, 5,
	//	2, 3, 4, 5, 6,
	//	3, 4, 5, 6, 7,
	//	4, 5, 6, 7, 8,
	//	5, 6, 7, 8, 9,
	//}

	tril := inputs.TriangleLower(math.Inf(-1))
	trilsm := tril.Softmax()
	mean := trilsm.Mean()

	tril.Forward()
	trilsm.Forward()
	mean.Forward()

	mean.ResetGrads(1.0)
	mean.Backward()
	trilsm.Backward()
	tril.Backward()

	fmt.Println(trilsm.StringData())
	fmt.Println(trilsm.StringGrad())
	fmt.Println(inputs.StringGrad())
}
