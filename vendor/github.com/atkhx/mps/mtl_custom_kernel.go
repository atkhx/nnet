package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation
#include "mtl_custom_kernel.h"
*/
import "C"
import (
	_ "embed"
	"unsafe"
)

//go:embed custom_kernel.metal
var customKernelFunctions string

func customKernelCreate(deviceID unsafe.Pointer) unsafe.Pointer {
	cKernelString := C.CString(customKernelFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return C.customKernelCreate(deviceID, cKernelString)
}

func customKernelCopy(kernelID, commandBufferID, dstBufferID, srcBufferID unsafe.Pointer, dstOffset, srcOffset, length int) {
	C.customKernelCopy(kernelID, commandBufferID, dstBufferID, srcBufferID, C.uint(dstOffset*4), C.uint(srcOffset*4), C.uint(length*4))
}

func customKernelFill(kernelID, commandBufferID, bufferID unsafe.Pointer, value float32, offset, length int) {
	C.customKernelFill(kernelID, commandBufferID, bufferID, C.float(value), C.uint(offset*4), C.uint(length*4))
}

func customKernelAdd(kernelID, commandBufferID, dstBufferID, srcBufferID unsafe.Pointer, dstOffset, srcOffset, length int) {
	C.customKernelAdd(kernelID, commandBufferID, dstBufferID, srcBufferID, C.uint(dstOffset*4), C.uint(srcOffset*4), C.uint(length*4))
}

func customKernelAddTo(kernelID, commandBufferID, dstBufferID, aBuffer, bBuffer unsafe.Pointer) {
	C.customKernelAddTo(kernelID, commandBufferID, dstBufferID, aBuffer, bBuffer)
}

func customKernelAddScalar(kernelID, commandBufferID, dstBufferID unsafe.Pointer, value float32) {
	C.customKernelAddScalar(kernelID, commandBufferID, dstBufferID, C.float(value))
}

func customKernelMul(kernelID, commandBufferID, dstBufferID, srcBufferID unsafe.Pointer, dstOffset, srcOffset, length int) {
	C.customKernelMul(kernelID, commandBufferID, dstBufferID, srcBufferID, C.uint(dstOffset*4), C.uint(srcOffset*4), C.uint(length*4))
}

func customKernelReLU(kernelID, commandBufferID, dstBufferID, srcBufferID unsafe.Pointer) {
	C.customKernelReLU(kernelID, commandBufferID, dstBufferID, srcBufferID)
}

func customKernelReLUBackward(kernelID, commandBufferID, dstBufferID, srcBufferID, maskBufferID unsafe.Pointer) {
	C.customKernelReLUBwd(kernelID, commandBufferID, dstBufferID, srcBufferID, maskBufferID)
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

func customKernelDropout(
	kernelID,
	commandBufferID,
	dstBufferID,
	srcBufferID,
	mskBufferID unsafe.Pointer,
	probability float32,
) {
	C.customKernelDropout(
		kernelID,
		commandBufferID,
		dstBufferID,
		srcBufferID,
		mskBufferID,
		C.float(probability),
	)
}

func customKernelDropoutBwd(
	kernelID,
	commandBufferID,
	dstBufferID,
	srcBufferID,
	mskBufferID unsafe.Pointer,
	probability float32,
) {
	C.customKernelDropoutBwd(
		kernelID,
		commandBufferID,
		dstBufferID,
		srcBufferID,
		mskBufferID,
		C.float(probability),
	)
}

func customKernelUpdateWithAdam(
	kernelID,
	commandBufferID,

	dataBufferID,
	gradBufferID,
	mBufferID,
	vBufferID unsafe.Pointer,

	beta1,
	beta2,
	beta1powIterationLR,
	beta2powIteration float32,
) {
	C.customKernelUpdateWithAdam(
		kernelID,
		commandBufferID,

		dataBufferID,
		gradBufferID,
		mBufferID,
		vBufferID,

		C.float(beta1),
		C.float(beta2),
		C.float(beta1powIterationLR),
		C.float(beta2powIteration),
	)
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

func customKernelCrossEntropyPos(
	kernelID,
	commandBufferID,
	dstBufferID,
	srcBufferID,
	smxBufferID,
	sumBufferID,
	tgtBufferID unsafe.Pointer,
	chunkSize int,
) {
	C.customKernelCrossEntropyPos(
		kernelID,
		commandBufferID,
		dstBufferID,
		srcBufferID,
		smxBufferID,
		sumBufferID,
		tgtBufferID,
		C.uint(chunkSize),
	)
}

func customKernelCrossEntropyPosBwd(
	kernelID,
	commandBufferID,
	oGrad,
	aGrad,
	tgtBufferID,
	smxBufferID unsafe.Pointer,
	chunkSize int,
) {
	C.customKernelCrossEntropyPosBwd(
		kernelID,
		commandBufferID,
		oGrad,
		aGrad,
		tgtBufferID,
		smxBufferID,
		C.uint(chunkSize),
	)
}

func customKernelRMSNorm(
	kernelID,
	commandBufferID,
	dstBufferID,
	srcBufferID,
	sumBufferID unsafe.Pointer,
	chunkSize int,
) {
	C.customKernelRMSNorm(
		kernelID,
		commandBufferID,
		dstBufferID,
		srcBufferID,
		sumBufferID,
		C.uint(chunkSize),
	)
}
