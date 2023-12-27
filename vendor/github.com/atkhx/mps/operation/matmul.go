package operation

import (
	"github.com/atkhx/mps"
)

func NewOpMatrixMultiply(
	device *mps.MTLDevice,

	aDataBuffer *mps.MTLBuffer,
	aGradBuffer *mps.MTLBuffer,

	bDataBuffer *mps.MTLBuffer,
	bGradBuffer *mps.MTLBuffer,

	cDataBuffer *mps.MTLBuffer,
	cGradBuffer *mps.MTLBuffer,

	aWidth, aHeight, aDepth int,
	bWidth, bHeight, bDepth int,
	cWidth, cHeight, cDepth int,

	alpha float32,
) Operation {
	if aDepth == bDepth {
		return NewOpMatrixMultiplyEqual(
			device,
			aDataBuffer,
			aGradBuffer,
			bDataBuffer,
			bGradBuffer,
			cDataBuffer,
			cGradBuffer,
			aWidth, aHeight, aDepth,
			bWidth, bHeight, bDepth,
			cWidth, cHeight, cDepth,
			alpha,
		)
	}

	if bDepth == 1 {
		return NewOpMatrixMultiplyOnFlat(
			device,
			aDataBuffer,
			aGradBuffer,
			bDataBuffer,
			bGradBuffer,
			cDataBuffer,
			cGradBuffer,
			aWidth, aHeight, aDepth,
			bWidth, bHeight, bDepth,
			cWidth, cHeight, cDepth,
			alpha,
		)
	}

	if aDepth == 1 {
		return NewOpMatrixMultiplyFlat(
			device,
			aDataBuffer,
			aGradBuffer,
			bDataBuffer,
			bGradBuffer,
			cDataBuffer,
			cGradBuffer,
			aWidth, aHeight, aDepth,
			bWidth, bHeight, bDepth,
			cWidth, cHeight, cDepth,
			alpha,
		)
	}

	panic("not implemented case aDepth != bDepth")
}
