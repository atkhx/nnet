package num

import (
	"fmt"
	"testing"
)

func TestData_TriangleLowerMatrixMultiply(t *testing.T) {
	aData := New(NewDims(3, 3))
	aData.Data = Float64s{
		1, 2, 3,
		2, 3, 4,
		4, 5, 6,
	}
	bData := New(NewDims(3, 3))
	bData.Data = Float64s{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
	}
	cData := aData.TriangleLowerMatrixMultiply(bData)
	cMean := cData.Mean()

	cData.Forward()
	cMean.Forward()

	cMean.ResetGrads(1.0)
	cMean.Backward()
	cData.Backward()

	fmt.Println(cData.StringData())
	fmt.Println(cData.StringGrad())
}
