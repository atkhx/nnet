package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/framework"
)

func NewOpMatrixMultiplyFlat(
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
) *OpMatrixMultiplyFlat {
	if aDepth != 1 {
		panic("aDepth != 1")
	}

	op := &OpMatrixMultiplyFlat{}

	batchSize := bDepth

	op.aDataM = aDataBuffer.CreateMatrixBatch(aWidth, aHeight, batchSize, 0, 0).MatrixID
	op.bDataM = bDataBuffer.CreateMatrixBatch(bWidth, bHeight, batchSize, bWidth*bHeight, 0).MatrixID
	op.cDataM = cDataBuffer.CreateMatrixBatch(cWidth, cHeight, batchSize, cWidth*cHeight, 0).MatrixID

	op.aGradM = aGradBuffer.CreateMatrix(aWidth, aHeight, 0).MatrixID
	op.bGradM = bGradBuffer.CreateMatrixBatch(bWidth, bHeight, batchSize, bWidth*bHeight, 0).MatrixID
	op.cGradM = cGradBuffer.CreateMatrixBatch(cWidth, cHeight, batchSize, cWidth*cHeight, 0).MatrixID

	op.calcCDataKernelID = device.CreateMatrixMultiplyKernel(aHeight, bWidth, aWidth, alpha, 0.0, false, false)
	op.calcBGradKernelID = device.CreateMatrixMultiplyKernel(bHeight, bWidth, cHeight, alpha, 1.0, true, false)
	op.calcAGradKernelID = device.CreateMatrixMultiplyKernel(aHeight, aWidth, cWidth, alpha, 1.0, false, true)

	op.bDataMs = make([]unsafe.Pointer, 0, batchSize)
	op.cGradMs = make([]unsafe.Pointer, 0, batchSize)

	for i := 0; i < batchSize; i++ {
		op.bDataMs = append(op.bDataMs, bDataBuffer.CreateMatrix(bWidth, bHeight, i*bWidth*bHeight).MatrixID)
		op.cGradMs = append(op.cGradMs, cGradBuffer.CreateMatrix(cWidth, cHeight, i*cWidth*cHeight).MatrixID)
	}

	return op
}

type OpMatrixMultiplyFlat struct {
	calcCDataKernelID unsafe.Pointer
	calcAGradKernelID unsafe.Pointer
	calcBGradKernelID unsafe.Pointer

	aDataM, bDataM, cDataM unsafe.Pointer
	aGradM, bGradM, cGradM unsafe.Pointer

	cGradMs, bDataMs []unsafe.Pointer
}

func (op *OpMatrixMultiplyFlat) Forward(b *mps.MTLCommandBuffer) {
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

func (op *OpMatrixMultiplyFlat) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		framework.MPSMatrixMultiplicationEncode(
			b.ID,
			op.calcBGradKernelID,
			op.aDataM,
			op.cGradM,
			op.bGradM,
		)

		for i := 0; i < len(op.cGradMs); i++ {
			framework.MPSMatrixMultiplicationEncode(
				b.ID,
				op.calcAGradKernelID,
				op.cGradMs[i],
				op.bDataMs[i],
				op.aGradM,
			)
		}
	})
}
