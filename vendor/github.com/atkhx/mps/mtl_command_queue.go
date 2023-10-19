package mps

import "C"
import "unsafe"

func NewMTLCommandQueue(device *MTLDevice) *MTLCommandQueue {
	return &MTLCommandQueue{
		queueID: mtlCommandQueueCreate(device.deviceID),
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
		mtlCommandQueueRelease(queue.queueID)
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
