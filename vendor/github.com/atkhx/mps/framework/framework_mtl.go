package framework

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders  -framework Foundation
#include "framework_mtl.h"
*/
import "C"
import (
	"unsafe"
)

// MTLDevice

func MTLDeviceCreate() unsafe.Pointer {
	return unsafe.Pointer(C.mtlDeviceCreate())
}

func MTLDeviceRelease(deviceID unsafe.Pointer) {
	C.mtlDeviceRelease(deviceID)
}

// MTLCommandQueue

func MTLCommandQueueCreate(deviceID unsafe.Pointer) unsafe.Pointer {
	return C.mtlCommandQueueCreate(deviceID)
}

func MTLCommandQueueRelease(commandBufferID unsafe.Pointer) {
	C.mtlCommandQueueRelease(commandBufferID)
}

// MTLCommandBuffer

func MTLCommandBufferCreate(commandQueueID unsafe.Pointer) unsafe.Pointer {
	return C.mtlCommandBufferCreate(commandQueueID)
}

func MTLCommandBufferRelease(commandBufferID unsafe.Pointer) {
	C.mtlCommandBufferRelease(commandBufferID)
}

func MTLCommandBufferCommitAndWaitUntilCompleted(commandBufferID unsafe.Pointer) {
	C.mtlCommandBufferCommitAndWaitUntilCompleted(commandBufferID)
}

// MTLBuffer

func MTLBufferCreateWithBytes(deviceID unsafe.Pointer, data []float32) unsafe.Pointer {
	return C.mtlBufferCreateWithBytes(deviceID, (*C.float)(unsafe.Pointer(&data[0])), C.ulong(len(data)))
}

func MTLBufferCreateWithLength(deviceID unsafe.Pointer, bfLength int) unsafe.Pointer {
	return C.mtlBufferCreateWithLength(deviceID, C.ulong(bfLength))
}

func MTLBufferGetContents(bufferID unsafe.Pointer) unsafe.Pointer {
	return unsafe.Pointer(C.mtlBufferGetContents(bufferID))
}

const maxBufferSize = 32 * 1024 * 1024 * 1024

func MTLBufferGetContentsFloats(bufferID unsafe.Pointer, length int) []float32 {
	contents := MTLBufferGetContents(bufferID)

	byteSlice := (*[maxBufferSize]byte)(contents)[:length:length]
	float32Slice := *(*[]float32)(unsafe.Pointer(&byteSlice))

	return float32Slice
}

func MTLBufferRelease(bufferID unsafe.Pointer) {
	C.mtlBufferRelease(bufferID)
}

func MTLCopyToBuffer(deviceID, srcBufferID, dstBufferID unsafe.Pointer, size int) {
	C.mtlCopyToBuffer(deviceID, srcBufferID, dstBufferID, C.ulong(size))
}

func MTLCopyToBufferWithCommandBuffer(commandBufferID, srcBufferID, dstBufferID unsafe.Pointer, size int) {
	C.mtlCopyToBufferWithCommandBuffer(commandBufferID, srcBufferID, dstBufferID, C.ulong(size))
}
