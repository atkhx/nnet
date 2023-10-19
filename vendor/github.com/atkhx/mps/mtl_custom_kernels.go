package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation
#include "mtl_custom_kernels.h"
*/
import "C"
import (
	_ "embed"
	"unsafe"
)

//go:embed krn_mtl_buffer_fill.metal
var kernelFill string

func customKernelFillCreate(deviceID unsafe.Pointer) unsafe.Pointer {
	cKernelString := C.CString(kernelFill)
	defer C.free(unsafe.Pointer(cKernelString))
	return C.customKernelFillCreate(deviceID, cKernelString)
}

func customKernelFill(kernelID, commandBufferID, bufferID unsafe.Pointer, value float32) {
	C.customKernelFill(kernelID, commandBufferID, bufferID, C.float(value))
}

func customKernelFillPart(kernelID, commandBufferID, bufferID unsafe.Pointer, offset, length int, value float32) {
	C.customKernelFillPart(kernelID, commandBufferID, bufferID, C.uint(offset*4), C.uint(length*4), C.float(value))
}

//go:embed krn_mtl_buffer_relu_fwd.metal
var kernelReLUFwd string

func customKernelReLUForwardCreate(deviceID unsafe.Pointer) unsafe.Pointer {
	cKernelString := C.CString(kernelReLUFwd)
	defer C.free(unsafe.Pointer(cKernelString))
	return C.customKernelReLUFwdCreate(deviceID, cKernelString)
}

func customKernelReLUForward(kernelID, commandBufferID, dstBufferID, srcBufferID unsafe.Pointer) {
	C.customKernelReLUFwd(kernelID, commandBufferID, dstBufferID, srcBufferID)
}

//go:embed krn_mtl_buffer_relu_bwd.metal
var kernelReLUBwd string

func customKernelReLUBackwardCreate(deviceID unsafe.Pointer) unsafe.Pointer {
	cKernelString := C.CString(kernelReLUBwd)
	defer C.free(unsafe.Pointer(cKernelString))
	return C.customKernelReLUBwdCreate(deviceID, cKernelString)
}

func customKernelReLUBackward(kernelID, commandBufferID, dstBufferID, srcBufferID, maskBufferID unsafe.Pointer) {
	C.customKernelReLUBwd(kernelID, commandBufferID, dstBufferID, srcBufferID, maskBufferID)
}

//go:embed krn_mtl_buffer_mul.metal
var kernelMul string

func customKernelMulCreate(deviceID unsafe.Pointer) unsafe.Pointer {
	cKernelString := C.CString(kernelMul)
	defer C.free(unsafe.Pointer(cKernelString))
	return C.customKernelMulCreate(deviceID, cKernelString)
}

func customKernelMul(kernelID, commandBufferID, dstBufferID, srcBufferID unsafe.Pointer) {
	C.customKernelMul(kernelID, commandBufferID, dstBufferID, srcBufferID)
}

//go:embed krn_mtl_buffer_dropout.metal
var kernelDropout string

func customKernelDropoutCreate(deviceID unsafe.Pointer) unsafe.Pointer {
	cKernelString := C.CString(kernelDropout)
	defer C.free(unsafe.Pointer(cKernelString))
	return C.customKernelDropoutCreate(deviceID, cKernelString)
}

func customKernelDropout(
	kernelID,
	commandBufferID,
	dstBufferID,
	srcBufferID,
	maskOutBufferID unsafe.Pointer,
	probability float32,
) {
	C.customKernelDropout(
		kernelID,
		commandBufferID,
		dstBufferID,
		srcBufferID,
		maskOutBufferID,
		C.float(probability),
	)
}

//go:embed krn_mtl_buffer_softmax.metal
var kernelSoftmax string

func customKernelSoftmaxForwardCreate(deviceID unsafe.Pointer) unsafe.Pointer {
	cKernelString := C.CString(kernelSoftmax)
	defer C.free(unsafe.Pointer(cKernelString))
	return C.customKernelSoftmaxCreate(deviceID, cKernelString)
}

func customKernelSoftmaxForward(
	kernelID,
	commandBufferID,
	dstBufferID,
	srcBufferID,
	sumOutBufferID unsafe.Pointer,
	colsCount, rowsCount, offset int,
) {
	C.customKernelSoftmax(
		kernelID,
		commandBufferID,
		dstBufferID,
		srcBufferID,
		sumOutBufferID,
		C.uint(colsCount),
		C.uint(rowsCount),
		C.uint(offset*4),
	)
}

//go:embed krn_mtl_buffer_softmax_tril_fwd.metal
var kernelSoftmaxTrilFwd string

func customKernelSoftmaxTrilForwardCreate(deviceID unsafe.Pointer) unsafe.Pointer {
	cKernelString := C.CString(kernelSoftmaxTrilFwd)
	defer C.free(unsafe.Pointer(cKernelString))
	return C.customKernelSoftmaxTrilFwdCreate(deviceID, cKernelString)
}

func customKernelSoftmaxTrilFwdCreate(
	kernelID,
	commandBufferID,
	dstBufferID,
	srcBufferID unsafe.Pointer,

	colsCount,
	rowsCount,
	offset int,
) {
	C.customKernelSoftmaxTrilFwd(
		kernelID,
		commandBufferID,
		dstBufferID,
		srcBufferID,
		C.uint(colsCount),
		C.uint(rowsCount),
		C.uint(offset*4),
	)
}

//go:embed krn_mtl_buffer_softmax_tril_bwd.metal
var kernelSoftmaxTrilBwd string

func customKernelSoftmaxTrilBackwardCreate(deviceID unsafe.Pointer) unsafe.Pointer {
	cKernelString := C.CString(kernelSoftmaxTrilBwd)
	defer C.free(unsafe.Pointer(cKernelString))
	return C.customKernelSoftmaxTrilBwdCreate(deviceID, cKernelString)
}

func customKernelSoftmaxTrilBackward(
	kernelID,
	commandBufferID,
	dstBufferID,
	srcBufferID,
	softmaxBufferID unsafe.Pointer,
	colsCount, rowsCount, offset int,
) {
	C.customKernelSoftmaxTrilBwd(
		kernelID,
		commandBufferID,
		dstBufferID,
		srcBufferID,
		softmaxBufferID,
		C.uint(colsCount),
		C.uint(rowsCount),
		C.uint(offset*4),
	)
}
