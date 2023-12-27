package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/framework"
)

func NewOpMatrixMultiplyOnFlat(
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
) *OpMatrixMultiplyOnFlat {
	if bDepth != 1 {
		panic("bDepth != 1")
	}

	op := &OpMatrixMultiplyOnFlat{}

	batchSize := aDepth

	op.aDataM = aDataBuffer.CreateMatrixBatch(aWidth, aHeight, batchSize, aWidth*aHeight, 0).MatrixID
	op.bDataM = bDataBuffer.CreateMatrixBatch(bWidth, bHeight, batchSize, 0, 0).MatrixID
	op.cDataM = cDataBuffer.CreateMatrixBatch(cWidth, cHeight, batchSize, cWidth*cHeight, 0).MatrixID

	op.aGradM = aGradBuffer.CreateMatrixBatch(aWidth, aHeight, batchSize, aWidth*aHeight, 0).MatrixID
	op.bGradM = bGradBuffer.CreateMatrix(bWidth, bHeight, 0).MatrixID
	op.cGradM = cGradBuffer.CreateMatrixBatch(cWidth, cHeight, batchSize, cWidth*cHeight, 0).MatrixID

	op.aDataMBig = aDataBuffer.CreateMatrix(aWidth, aHeight*aDepth, 0).MatrixID
	op.cGradMBig = cGradBuffer.CreateMatrix(cWidth, cHeight*cDepth, 0).MatrixID

	op.calcCDataKernelID = device.CreateMatrixMultiplyKernel(aHeight, bWidth, aWidth, alpha, 0.0, false, false)
	op.calcAGradKernelID = device.CreateMatrixMultiplyKernel(aHeight, aWidth, cWidth, alpha, 1.0, false, true)
	op.calcBGradKernelID = device.CreateMatrixMultiplyKernel(bHeight, bWidth, aHeight*aDepth, alpha, 1.0, true, false)

	return op
}

type OpMatrixMultiplyOnFlat struct {
	calcCDataKernelID unsafe.Pointer
	calcAGradKernelID unsafe.Pointer
	calcBGradKernelID unsafe.Pointer

	aDataM, bDataM, cDataM unsafe.Pointer
	aGradM, bGradM, cGradM unsafe.Pointer

	aDataMBig, cGradMBig unsafe.Pointer
}

func (op *OpMatrixMultiplyOnFlat) Forward(b *mps.MTLCommandBuffer) {
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

func (op *OpMatrixMultiplyOnFlat) Backward(b *mps.MTLCommandBuffer) {
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
			op.aDataMBig,
			op.cGradMBig,
			op.bGradM,
		)
	})
}
