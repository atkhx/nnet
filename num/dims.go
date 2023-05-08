package num

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

func (dims Dims) GetDimsByMax(bDims Dims) (cDims Dims) {
	return Dims{
		W: max(dims.W, bDims.W),
		H: max(dims.H, bDims.H),
		D: max(dims.D, bDims.D),
	}
}

type Steps struct {
	aW, aH, aD int
	bW, bH, bD int
}

func (dims Dims) GetBroadCastSteps(bDims Dims) (steps Steps) {
	steps = Steps{aW: 1, aH: 1, aD: 1, bW: 1, bH: 1, bD: 1}

	if dims.W != bDims.W {
		switch {
		case dims.W == 1:
			steps.aW = 0
		case bDims.W == 1:
			steps.bW = 0
		default:
			panic("A & B: width must be equal or one of them must be 1")
		}
	}

	if dims.H != bDims.H {
		switch {
		case dims.H == 1:
			steps.aH = 0
		case bDims.H == 1:
			steps.bH = 0
		default:
			panic("A & B: height must be equal or one of them must be 1")
		}
	}

	if dims.D != bDims.D {
		switch {
		case dims.D == 1:
			steps.aD = 0
		case bDims.D == 1:
			steps.bD = 0
		default:
			panic("A & B: dept must be equal or one of them must be 1")
		}
	}

	return
}
