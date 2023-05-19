package num

import (
	"testing"
)

func TestData_VarianceByRows(t *testing.T) {
	beta := New(NewDims(3))
	gamma := New(NewDims(3))
	gamma.Data.Fill(1)

	input := New(NewDims(3, 2))
	input.Data = Float64s{
		1.0, 0.5, 0.1,
		0.7, -0.3, 0.1,
	}

	lnorm := input.LNorm(gamma, beta)
	lnorm.Forward()

	mean := lnorm.Mean()
	mean.Forward()

	mean.ResetGrads(1.0)
	mean.Backward()
	lnorm.Backward()
}
