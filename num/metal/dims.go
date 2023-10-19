package metal

import "fmt"

const (
	dimsWidthIdx  = 0
	dimsHeightIdx = 1
	dimsDepthIdx  = 2
)

func NewDims(dims ...int) Dims {
	if len(dims) > 3 {
		panic("to much dimensions")
	}

	allDims := []int{1, 1, 1}
	copy(allDims, dims)

	return Dims{
		W: allDims[dimsWidthIdx],
		H: allDims[dimsHeightIdx],
		D: allDims[dimsDepthIdx],
	}
}

type Dims struct {
	W, H, D int
}

func (dims Dims) Size() int {
	return dims.W * dims.H * dims.D
}

func (dims Dims) IsEqual(bDims Dims) bool {
	return dims.W == bDims.W && dims.H == bDims.H && dims.D == bDims.D
}

func (dims Dims) MustBeEqual(bDims Dims) {
	if !dims.IsEqual(bDims) {
		panic("dimensions must be equal: " + fmt.Sprintf("expected: %v, actual %v", dims, bDims))
	}
}

func (dims Dims) GetMax(bDims Dims) (cDims Dims) {
	return Dims{
		W: max(dims.W, bDims.W),
		H: max(dims.H, bDims.H),
		D: max(dims.D, bDims.D),
	}
}
