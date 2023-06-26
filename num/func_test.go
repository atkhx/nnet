package num

import (
	"fmt"
	"testing"
)

func Test_transpose(t *testing.T) {
	a := New(NewDims(3, 2, 3))
	a.Data = Float64s{
		1, 2, 3,
		4, 5, 6,

		10, 11, 12,
		13, 14, 15,

		20, 21, 22,
		23, 24, 25,
	}

	fmt.Println(a.StringData())

	b := transpose(3, 2, a.Data)
	fmt.Println(b.String(NewDims(2, 3, 3)))
}
