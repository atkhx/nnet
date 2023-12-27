package mps

import (
	"unsafe"

	"github.com/atkhx/mps/framework"
)

const maxBufferSize = 32 * 1024 * 1024 * 1024

func newMTLBufferWithBytes(device *MTLDevice, data []float32) *MTLBuffer {
	bufferID := framework.MTLBufferCreateWithBytes(device.DeviceID, data)
	contents := framework.MTLBufferGetContents(bufferID)
	bfLength := len(data)

	byteSlice := (*[maxBufferSize]byte)(contents)[:bfLength:bfLength]
	float32Slice := *(*[]float32)(unsafe.Pointer(&byteSlice))

	return &MTLBuffer{
		BufferID: bufferID,
		device:   device,
		contents: float32Slice,
		Length:   bfLength,
	}
}

func newMTLBufferWithLength(device *MTLDevice, bfLength int) *MTLBuffer {
	bufferID := framework.MTLBufferCreateWithLength(device.DeviceID, bfLength)
	contents := framework.MTLBufferGetContents(bufferID)

	byteSlice := (*[maxBufferSize]byte)(contents)[:bfLength:bfLength]
	float32Slice := *(*[]float32)(unsafe.Pointer(&byteSlice))

	return &MTLBuffer{
		BufferID: bufferID,
		device:   device,
		contents: float32Slice,
		Length:   bfLength,
	}
}

type MTLBuffer struct {
	BufferID unsafe.Pointer
	device   *MTLDevice
	contents []float32
	Length   int
	released bool
}

func (buffer *MTLBuffer) CreateMatrix(cols, rows, offset int) *MPSMatrix {
	return NewMPSMatrix(buffer.BufferID, cols, rows, 1, 0, offset)
}

func (buffer *MTLBuffer) CreateMatrixBatch(cols, rows, batchSize, batchStride, offset int) *MPSMatrix {
	return NewMPSMatrix(buffer.BufferID, cols, rows, batchSize, batchStride, offset)
}

func (buffer *MTLBuffer) GetData() []float32 {
	return buffer.contents
}

func (buffer *MTLBuffer) Release() {
	if !buffer.released {
		framework.MTLBufferRelease(buffer.BufferID)
		buffer.released = true
	}
}
