package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics
#include "metal.h"
*/
import "C"
import (
	"unsafe"
)

// MTLDevice

func mtlDeviceCreate() unsafe.Pointer {
	return unsafe.Pointer(C.mtlDeviceCreate())
}

func mtlDeviceRelease(deviceID unsafe.Pointer) {
	C.mtlDeviceRelease(deviceID)
}

// MTLCommandQueue

func mtlCommandQueueCreate(deviceID unsafe.Pointer) unsafe.Pointer {
	return C.mtlCommandQueueCreate(deviceID)
}

func mtlCommandQueueRelease(commandBufferID unsafe.Pointer) {
	C.mtlCommandQueueRelease(commandBufferID)
}

// MTLCommandBuffer

func mtlCommandBufferCreate(commandQueueID unsafe.Pointer) unsafe.Pointer {
	return C.mtlCommandBufferCreate(commandQueueID)
}

func mtlCommandBufferRelease(commandBufferID unsafe.Pointer) {
	C.mtlCommandBufferRelease(commandBufferID)
}

func mtlCommandBufferCommitAndWaitUntilCompleted(commandBufferID unsafe.Pointer) {
	C.mtlCommandBufferCommitAndWaitUntilCompleted(commandBufferID)
}

// MTLBuffer

func mtlBufferCreateCreateWithBytes(deviceID unsafe.Pointer, data []float32) unsafe.Pointer {
	return C.mtlBufferCreateCreateWithBytes(deviceID, (*C.float)(unsafe.Pointer(&data[0])), C.ulong(len(data)))
}

func mtlBufferCreateWithLength(deviceID unsafe.Pointer, bfLength int) unsafe.Pointer {
	return C.mtlBufferCreateWithLength(deviceID, C.ulong(bfLength))
}

func mtlBufferGetContents(bufferID unsafe.Pointer) unsafe.Pointer {
	return unsafe.Pointer(C.mtlBufferGetContents(bufferID))
}

func mtlBufferRelease(bufferID unsafe.Pointer) {
	C.mtlBufferRelease(bufferID)
}

// MPSMatrixDescriptor

func mpsMatrixDescriptorCreate(cols, rows int) unsafe.Pointer {
	return C.mpsMatrixDescriptorCreate(C.int(cols), C.int(rows))
}

func mpsMatrixDescriptorRelease(descriptorID unsafe.Pointer) {
	C.mpsMatrixDescriptorRelease(descriptorID)
}

// MPSMatrix

func mpsMatrixCreate(bufferID, descriptorID unsafe.Pointer, offset int) unsafe.Pointer {
	return C.mpsMatrixCreate(bufferID, descriptorID, C.int(offset))
}

func mpsMatrixRelease(matrixID unsafe.Pointer) {
	C.mpsMatrixRelease(matrixID)
}

func mpsMatrixMultiply(
	deviceID,
	commandBufferID,

	aMatrixID,
	bMatrixID,
	cMatrixID unsafe.Pointer,

	interiorColumns int,

	transposeLeft,
	transposeRight bool,

	alpha,
	beta float32,
) {
	C.mpsMatrixMultiply(
		deviceID,
		commandBufferID,

		aMatrixID,
		bMatrixID,
		cMatrixID,

		C.int(interiorColumns),

		C.float(alpha),
		C.float(beta),

		C.bool(transposeLeft),
		C.bool(transposeRight),
	)
}
