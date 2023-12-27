package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/framework"
)

func NewOpMatrixMultiplyEqual(
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
) *OpMatrixMultiplyEqual {
	if aDepth != bDepth {
		panic("aDepth != bDepth")
	}

	op := &OpMatrixMultiplyEqual{}

	batchSize := aDepth

	op.aDataM = aDataBuffer.CreateMatrixBatch(aWidth, aHeight, batchSize, aWidth*aHeight, 0).MatrixID
	op.aGradM = aGradBuffer.CreateMatrixBatch(aWidth, aHeight, batchSize, aWidth*aHeight, 0).MatrixID
	op.bDataM = bDataBuffer.CreateMatrixBatch(bWidth, bHeight, batchSize, bWidth*bHeight, 0).MatrixID
	op.bGradM = bGradBuffer.CreateMatrixBatch(bWidth, bHeight, batchSize, bWidth*bHeight, 0).MatrixID
	op.cDataM = cDataBuffer.CreateMatrixBatch(cWidth, cHeight, batchSize, cWidth*cHeight, 0).MatrixID
	op.cGradM = cGradBuffer.CreateMatrixBatch(cWidth, cHeight, batchSize, cWidth*cHeight, 0).MatrixID

	op.calcCDataKernelID = device.CreateMatrixMultiplyKernel(aHeight, bWidth, aWidth, alpha, 0.0, false, false)
	op.calcAGradKernelID = device.CreateMatrixMultiplyKernel(aHeight, aWidth, cWidth, alpha, 1.0, false, true)
	op.calcBGradKernelID = device.CreateMatrixMultiplyKernel(bHeight, bWidth, aHeight, alpha, 1.0, true, false)

	return op
}

type OpMatrixMultiplyEqual struct {
	calcCDataKernelID unsafe.Pointer
	calcAGradKernelID unsafe.Pointer
	calcBGradKernelID unsafe.Pointer

	aDataM unsafe.Pointer
	bDataM unsafe.Pointer
	cDataM unsafe.Pointer

	aGradM unsafe.Pointer
	cGradM unsafe.Pointer
	bGradM unsafe.Pointer
}

func (op *OpMatrixMultiplyEqual) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		framework.MPSMatrixMultiplicationEncode(
			b.ID,
			op.calcCDataKernelID,
			op.aDataM,
			op.bDataM,
			op.cDataM,
		)
	})
}

func (op *OpMatrixMultiplyEqual) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		framework.MPSMatrixMultiplicationEncode(
			b.ID,
			op.calcAGradKernelID,
			op.cGradM,
			op.bDataM,
			op.aGradM,
		)
		framework.MPSMatrixMultiplicationEncode(
			b.ID,
			op.calcBGradKernelID,
			op.aDataM,
			op.cGradM,
			op.bGradM,
		)
	})
}
