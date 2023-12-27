package mps

import "C"
import (
	"unsafe"

	"github.com/atkhx/mps/framework"
)

func NewMTLCommandQueue(device *MTLDevice) *MTLCommandQueue {
	return &MTLCommandQueue{
		queueID: framework.MTLCommandQueueCreate(device.DeviceID),
		device:  device,
	}
}

type MTLCommandQueue struct {
	queueID  unsafe.Pointer
	device   *MTLDevice
	released bool
	buffer   *MTLCommandBuffer
}

func (queue *MTLCommandQueue) Release() {
	if !queue.released {
		queue.buffer.Release()
		framework.MTLCommandQueueRelease(queue.queueID)
		queue.released = true
	}
}

func (queue *MTLCommandQueue) GetCommandBuffer() *MTLCommandBuffer {
	switch {
	case queue.buffer != nil && !queue.buffer.completed:
		return queue.buffer
	case queue.buffer != nil && queue.buffer.completed:
		queue.buffer.Release()
	}

	queue.buffer = NewMTLCommandBuffer(queue)
	return queue.buffer
}
