package num

import (
	"fmt"
	"testing"
)

func TestData_VarianceByRows(t *testing.T) {
	input := New(NewDims(3, 2))
	input.Data = Float64s{
		1.0, 0.5, 0.1,
		0.7, -0.3, 0.1,
	}

	xvar := input.VarianceByRows(nil)
	xvar.Forward()

	fmt.Println("xvar.data", xvar.StringData())

}
