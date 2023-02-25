package main

import (
	"fmt"

	"github.com/atkhx/nnet/data"
)

func main() {
	a := data.NewScalar(+2.0)
	b := data.NewScalar(-3.0)
	c := data.NewScalar(10.0)

	d := a.Mul(b).Add(c)

	fmt.Println("d", d)
	d.Backward()

	showGradsV := []*data.Scalar{a, b, c, d}
	showGradsL := []string{"a", "b", "c", "d"}

	for i, v := range showGradsV {
		fmt.Println(showGradsL[i], "Grad", v.Grad)
	}
}
