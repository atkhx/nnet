package mps

import "unsafe"

func NewMTLBufferWithBytes(device *MTLDevice, data []float32) *MTLBuffer {
	bufferID := mtlBufferCreateCreateWithBytes(device.deviceID, data)
	contents := mtlBufferGetContents(bufferID)
	bfLength := len(data)

	byteSlice := (*[1 << 30]byte)(contents)[:bfLength:bfLength]
	float32Slice := *(*[]float32)(unsafe.Pointer(&byteSlice))

	return &MTLBuffer{
		bufferID: bufferID,
		device:   device,
		contents: float32Slice,
		length:   bfLength,
	}
}

func NewMTLBufferWithLength(device *MTLDevice, bfLength int) *MTLBuffer {
	bufferID := mtlBufferCreateWithLength(device.deviceID, bfLength)
	contents := mtlBufferGetContents(bufferID)

	byteSlice := (*[1 << 30]byte)(contents)[:bfLength:bfLength]
	float32Slice := *(*[]float32)(unsafe.Pointer(&byteSlice))

	return &MTLBuffer{
		bufferID: bufferID,
		device:   device,
		contents: float32Slice,
		length:   bfLength,
	}
}

type MTLBuffer struct {
	bufferID unsafe.Pointer
	device   *MTLDevice
	contents []float32
	length   int
	released bool
}

func (device *MTLDevice) CreateNewBufferWithLength(bfLength int) *MTLBuffer {
	bufferID := mtlBufferCreateWithLength(device.deviceID, bfLength)
	contents := mtlBufferGetContents(bufferID)

	byteSlice := (*[1 << 30]byte)(contents)[:bfLength:bfLength]
	float32Slice := *(*[]float32)(unsafe.Pointer(&byteSlice))

	buffer := &MTLBuffer{
		bufferID: bufferID,
		device:   device,
		contents: float32Slice,
		length:   bfLength,
	}

	device.regSource(buffer)
	return buffer
}

func (buffer *MTLBuffer) CreateMatrix(cols, rows, offset int) *Matrix {
	return NewMatrix(buffer, cols, rows, offset)
}

func (buffer *MTLBuffer) GetData() []float32 {
	return buffer.contents
}

func (buffer *MTLBuffer) Release() {
	if !buffer.released {
		mtlBufferRelease(buffer.bufferID)
		buffer.released = true
	}
}
