package mps

import "C"
import (
	"context"
	"sync"
	"unsafe"

	"github.com/atkhx/mps/custom-kernel"
	"github.com/atkhx/mps/framework"
)

func ContextWithCommandBuffer(ctx context.Context, buffer *MTLCommandBuffer) context.Context {
	return context.WithValue(ctx, "MTLCommandBuffer", buffer)
}

func CommandBufferFromContext(ctx context.Context) *MTLCommandBuffer {
	return ctx.Value("MTLCommandBuffer").(*MTLCommandBuffer)
}

func NewMTLCommandBuffer(queue *MTLCommandQueue) *MTLCommandBuffer {
	return &MTLCommandBuffer{
		ID:       framework.MTLCommandBufferCreate(queue.queueID),
		deviceID: queue.device.DeviceID,
		device:   queue.device,
	}
}

type MTLCommandBuffer struct {
	ID       unsafe.Pointer
	deviceID unsafe.Pointer
	device   *MTLDevice

	uncommitted int64
	completed   bool
	released    bool

	mu sync.Mutex
}

func (b *MTLCommandBuffer) Release() {
	if !b.released {
		framework.MTLCommandBufferRelease(b.ID)
		b.released = true
		b.completed = true
	}
}

func (b *MTLCommandBuffer) Wait() {
	b.Exclusive(func() {
		if b.uncommitted > 0 {
			framework.MTLCommandBufferCommitAndWaitUntilCompleted(b.ID)
			b.completed = true
		}
		b.uncommitted = 0
	})
}

func (b *MTLCommandBuffer) Exclusive(operation func()) {
	b.mu.Lock()
	operation()
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) ClearMTLBuffer(buffer *MTLBuffer) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelFill(b.device.CustomKernels, b.ID, buffer.BufferID, 0.0, 0, buffer.Length)
	})
}

func (b *MTLCommandBuffer) FillMTLBuffer(buffer *MTLBuffer, value float32) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelFill(b.device.CustomKernels, b.ID, buffer.BufferID, value, 0, buffer.Length)
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
	b.Exclusive(func() {
		custom_kernel.CustomKernelUpdateWithAdam(
			b.device.CustomKernels, b.ID,
			dataBuffer.BufferID,
			gradBuffer.BufferID,
			mBuffer.BufferID,
			vBuffer.BufferID,
			beta1,
			beta2,
			beta1powIterationLR,
			beta2powIteration,
		)
	})
}
