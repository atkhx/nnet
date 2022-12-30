//go:build amd64 && !noasm

package floats

func Dot(sliceA, sliceB []float64) (dot float64)

func MultiplyTo(dst, src []float64, k float64)

func MultiplyAndAddTo(dst, src []float64, k float64)
