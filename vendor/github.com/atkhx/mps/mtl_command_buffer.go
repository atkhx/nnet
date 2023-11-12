package mps

import "C"
import (
	"context"
	"sync"
	"unsafe"
)

func ContextWithCommandBuffer(ctx context.Context, buffer *MTLCommandBuffer) context.Context {
	return context.WithValue(ctx, "MTLCommandBuffer", buffer)
}

func CommandBufferFromContext(ctx context.Context) *MTLCommandBuffer {
	return ctx.Value("MTLCommandBuffer").(*MTLCommandBuffer)
}

func NewMTLCommandBuffer(queue *MTLCommandQueue) *MTLCommandBuffer {
	return &MTLCommandBuffer{
		id:       mtlCommandBufferCreate(queue.queueID),
		deviceID: queue.device.deviceID,
		device:   queue.device,
	}
}

type MTLCommandBuffer struct {
	id       unsafe.Pointer
	deviceID unsafe.Pointer
	device   *MTLDevice

	uncommitted int64
	completed   bool
	released    bool

	mu sync.Mutex
}

func (b *MTLCommandBuffer) Release() {
	if !b.released {
		mtlCommandBufferRelease(b.id)
		b.released = true
		b.completed = true
	}
}

func (b *MTLCommandBuffer) Wait() {
	b.exclusive(func() {
		if b.uncommitted > 0 {
			mtlCommandBufferCommitAndWaitUntilCompleted(b.id)
			b.completed = true
		}
		b.uncommitted = 0
	})
}

func (b *MTLCommandBuffer) exclusive(operation func()) {
	b.mu.Lock()
	operation()
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) Copy(dst, src *MTLBuffer, dstOffset, srcOffset, length int) {
	b.exclusive(func() {
		customKernelCopy(b.device.customKernels, b.id, dst.bufferID, src.bufferID, dstOffset, srcOffset, length)
	})
}

func (b *MTLCommandBuffer) ClearMTLBuffer(buffer *MTLBuffer) {
	b.exclusive(func() {
		customKernelFill(b.device.customKernels, b.id, buffer.bufferID, 0.0, 0, buffer.length)
	})
}

func (b *MTLCommandBuffer) FillMTLBuffer(buffer *MTLBuffer, value float32) {
	b.exclusive(func() {
		customKernelFill(b.device.customKernels, b.id, buffer.bufferID, value, 0, buffer.length)
	})
}

func (b *MTLCommandBuffer) FillMTLBufferPart(buffer *MTLBuffer, value float32, offset, length int) {
	b.exclusive(func() {
		customKernelFill(b.device.customKernels, b.id, buffer.bufferID, value, offset, length)
	})
}

func (b *MTLCommandBuffer) Add(dst, src *MTLBuffer, dstOffset, srcOffset, length int) {
	b.exclusive(func() {
		customKernelAdd(b.device.customKernels, b.id, dst.bufferID, src.bufferID, dstOffset, srcOffset, length)
	})
}

func (b *MTLCommandBuffer) AddTo(dst, aBuffer, bBuffer *MTLBuffer) {
	b.exclusive(func() {
		customKernelAddTo(b.device.customKernels, b.id, dst.bufferID, aBuffer.bufferID, bBuffer.bufferID)
	})
}

func (b *MTLCommandBuffer) AddScalar(dst *MTLBuffer, value float32) {
	b.exclusive(func() {
		customKernelAddScalar(b.device.customKernels, b.id, dst.bufferID, value)
	})
}

func (b *MTLCommandBuffer) Mul(dst, src *MTLBuffer, dstOffset, srcOffset, length int) {
	b.exclusive(func() {
		customKernelMul(b.device.customKernels, b.id, dst.bufferID, src.bufferID, dstOffset, srcOffset, length)
	})
}

func (b *MTLCommandBuffer) ReLuMTLBuffer(dstBuffer, srcBuffer *MTLBuffer) {
	b.exclusive(func() {
		customKernelReLU(b.device.customKernels, b.id, dstBuffer.bufferID, srcBuffer.bufferID)
	})
}

func (b *MTLCommandBuffer) ReLuMTLBufferBwd(dstBuffer, srcBuffer, maskBuffer *MTLBuffer) {
	b.exclusive(func() {
		customKernelReLUBackward(b.device.customKernels, b.id, dstBuffer.bufferID, srcBuffer.bufferID, maskBuffer.bufferID)
	})
}

func (b *MTLCommandBuffer) SoftmaxBuffer(
	destinationBuffer *MTLBuffer,
	sourceBuffer *MTLBuffer,
	sumOutBuffer *MTLBuffer,
	colsCount, rowsCount, offset int,
) {
	b.exclusive(func() {
		customKernelSoftmaxForward(
			b.device.customKernels,
			b.id,
			destinationBuffer.bufferID,
			sourceBuffer.bufferID,
			sumOutBuffer.bufferID,
			colsCount,
			rowsCount,
			offset,
		)
	})
}

func (b *MTLCommandBuffer) DropoutBuffer(
	dstBuffer,
	srcBuffer,
	mskBuffer *MTLBuffer,
	probability float32,
) {
	b.exclusive(func() {
		customKernelDropout(
			b.device.customKernels,
			b.id,
			dstBuffer.bufferID,
			srcBuffer.bufferID,
			mskBuffer.bufferID,
			probability,
		)
	})
}

func (b *MTLCommandBuffer) DropoutBwdBuffer(
	dstBuffer,
	srcBuffer,
	mskBuffer *MTLBuffer,
	probability float32,
) {
	b.exclusive(func() {
		customKernelDropoutBwd(
			b.device.customKernels,
			b.id,
			dstBuffer.bufferID,
			srcBuffer.bufferID,
			mskBuffer.bufferID,
			probability,
		)
	})
}

func (b *MTLCommandBuffer) UpdateWithAdam(
	dataBuffer,
	gradBuffer,
	mBuffer,
	vBuffer *MTLBuffer,

	beta1,
	beta2,
	beta1powIterationLR,
	beta2powIteration float32,
) {
	b.exclusive(func() {
		customKernelUpdateWithAdam(
			b.device.customKernels, b.id,
			dataBuffer.bufferID,
			gradBuffer.bufferID,
			mBuffer.bufferID,
			vBuffer.bufferID,
			beta1,
			beta2,
			beta1powIterationLR,
			beta2powIteration,
		)
	})
}

// not refactored part

func (b *MTLCommandBuffer) SoftmaxBufferTril(
	destinationBuffer *MTLBuffer,
	sourceBuffer *MTLBuffer,
	colsCount, rowsCount, offset int,
) {
	b.exclusive(func() {
		customKernelSoftmaxTrilFwdCreate(
			b.device.customKernels,
			b.id,
			destinationBuffer.bufferID,
			sourceBuffer.bufferID,
			colsCount,
			rowsCount,
			offset,
		)
	})
}

func (b *MTLCommandBuffer) SoftmaxBufferTrilBwd(
	destinationBuffer *MTLBuffer,
	sourceBuffer *MTLBuffer,
	softmaxBuffer *MTLBuffer,
	colsCount, rowsCount, offset int,
) {
	b.exclusive(func() {
		customKernelSoftmaxTrilBackward(
			b.device.customKernels,
			b.id,
			destinationBuffer.bufferID,
			sourceBuffer.bufferID,
			softmaxBuffer.bufferID,
			colsCount,
			rowsCount,
			offset,
		)
	})
}

func (b *MTLCommandBuffer) CrossEntropyPos(
	dstBuffer *MTLBuffer,
	srcBuffer *MTLBuffer,
	smxBuffer *MTLBuffer,
	sumBuffer *MTLBuffer,
	tgtBuffer *MTLBuffer,
	chunkSize int,
) {
	b.exclusive(func() {
		customKernelCrossEntropyPos(
			b.device.customKernels,
			b.id,
			dstBuffer.bufferID,
			srcBuffer.bufferID,
			smxBuffer.bufferID,
			sumBuffer.bufferID,
			tgtBuffer.bufferID,
			chunkSize,
		)
	})
}

func (b *MTLCommandBuffer) CrossEntropyPosBwd(
	oGrad *MTLBuffer,
	aGrad *MTLBuffer,
	tgtBuffer *MTLBuffer,
	smxBuffer *MTLBuffer,
	chunkSize int,
) {
	b.exclusive(func() {
		customKernelCrossEntropyPosBwd(
			b.device.customKernels,
			b.id,
			oGrad.bufferID,
			aGrad.bufferID,
			tgtBuffer.bufferID,
			smxBuffer.bufferID,
			chunkSize,
		)
	})
}

func (b *MTLCommandBuffer) RMSNorm(
	dstBuffer *MTLBuffer,
	srcBuffer *MTLBuffer,
	sumBuffer *MTLBuffer,
	chunkSize int,
) {
	b.exclusive(func() {
		customKernelRMSNorm(
			b.device.customKernels,
			b.id,
			dstBuffer.bufferID,
			srcBuffer.bufferID,
			sumBuffer.bufferID,
			chunkSize,
		)
	})
}

func (b *MTLCommandBuffer) MatrixMultiplyAB(aM, bM, cM *Matrix, alpha, beta float32) {
	b.MatrixMultiply(aM, bM, cM, aM.cols, false, false, alpha, beta)
}

func (b *MTLCommandBuffer) MatrixMultiplyATB(aM, bM, cM *Matrix, alpha, beta float32) {
	b.MatrixMultiply(aM, bM, cM, aM.cols, false, true, alpha, beta)
}

func (b *MTLCommandBuffer) MatrixMultiplyTAB(aM, bM, cM *Matrix, alpha, beta float32) {
	b.MatrixMultiply(aM, bM, cM, bM.rows, true, false, alpha, beta)
}

func (b *MTLCommandBuffer) MatrixMultiply(aM, bM, cM *Matrix, iC int, aT, bT bool, alpha, beta float32) {
	b.exclusive(func() {
		mpsMatrixMultiply(b.deviceID, b.id, aM.matrixID, bM.matrixID, cM.matrixID, iC, aT, bT, alpha, beta)
	})
}

func (b *MTLCommandBuffer) MatrixRandom(randomizer *MatrixRandomMTGP32, aM *Matrix) {
	b.exclusive(func() {
		mpsMatrixRandom(randomizer.id, b.id, aM.matrixID)
	})
}
