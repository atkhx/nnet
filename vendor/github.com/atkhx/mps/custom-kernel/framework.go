package custom_kernel

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation
#include "framework.h"
*/
import "C"
import (
	_ "embed"
	"unsafe"
)

//go:embed kernel.metal
var customKernelFunctions string

func CustomKernelCreate(deviceID unsafe.Pointer) unsafe.Pointer {
	cKernelString := C.CString(customKernelFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return C.customKernelCreate(deviceID, cKernelString)
}

func CustomKernelCopy(kernelID, commandBufferID, dstBufferID, srcBufferID unsafe.Pointer, dstOffset, srcOffset, length int) {
	C.customKernelCopy(kernelID, commandBufferID, dstBufferID, srcBufferID, C.uint(dstOffset*4), C.uint(srcOffset*4), C.uint(length*4))
}

func CustomKernelCopyWHD(kernelID, commandBufferID, dstBufferID, srcBufferID unsafe.Pointer, W, H, D int) {
	C.customKernelCopyWHD(kernelID, commandBufferID, dstBufferID, srcBufferID, C.uint(W), C.uint(H), C.uint(D))
}

func CustomKernelFill(kernelID, commandBufferID, bufferID unsafe.Pointer, value float32, offset, length int) {
	C.customKernelFill(kernelID, commandBufferID, bufferID, C.float(value), C.uint(offset*4), C.uint(length*4))
}

func CustomKernelAdd(kernelID, commandBufferID, dstBufferID, srcBufferID unsafe.Pointer, dstOffset, srcOffset, length int) {
	C.customKernelAdd(kernelID, commandBufferID, dstBufferID, srcBufferID, C.uint(dstOffset*4), C.uint(srcOffset*4), C.uint(length*4))
}

func CustomKernelAddTo(kernelID, commandBufferID, dstBufferID, aBuffer, bBuffer unsafe.Pointer) {
	C.customKernelAddTo(kernelID, commandBufferID, dstBufferID, aBuffer, bBuffer)
}

func CustomKernelAddToWHD(kernelID, commandBufferID, dstBufferID, aBuffer, bBuffer unsafe.Pointer, K float32, W, H, D int) {
	C.customKernelAddToWHD(kernelID, commandBufferID, dstBufferID, aBuffer, bBuffer, C.float(K), C.uint(W), C.uint(H), C.uint(D))
}

func CustomKernelAddToWHDBwd(kernelID, commandBufferID, aGrad, bGrad, oGrad unsafe.Pointer, W, H, D int) {
	C.customKernelAddToWHDBwd(kernelID, commandBufferID, aGrad, bGrad, oGrad, C.uint(W), C.uint(H), C.uint(D))
}

func CustomKernelAddScalar(kernelID, commandBufferID, dstBufferID unsafe.Pointer, value float32) {
	C.customKernelAddScalar(kernelID, commandBufferID, dstBufferID, C.float(value))
}

func CustomKernelMul(kernelID, commandBufferID, dstBufferID, srcBufferID unsafe.Pointer, dstOffset, srcOffset, length int) {
	C.customKernelMul(kernelID, commandBufferID, dstBufferID, srcBufferID, C.uint(dstOffset*4), C.uint(srcOffset*4), C.uint(length*4))
}

func CustomKernelSoftmaxForward(
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

func CustomKernelUpdateWithAdam(
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

func CustomKernelSoftmaxTrilFwdCreate(
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

func CustomKernelSoftmaxTrilBackward(
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

func CustomKernelNLLByPos(
	kernelID,
	commandBufferID,
	dstBufferID,
	smxBufferID,
	tgtBufferID unsafe.Pointer,
	chunkSize int,
) {
	C.customKernelNLLByPos(
		kernelID,
		commandBufferID,
		dstBufferID,
		smxBufferID,
		tgtBufferID,
		C.uint(chunkSize),
	)
}

func CustomKernelNLLByPosBwd(
	kernelID,
	commandBufferID,
	oGrad,
	aGrad,
	tgtBufferID,
	smxBufferID unsafe.Pointer,
	chunkSize int,
) {
	C.customKernelNLLByPosBwd(
		kernelID,
		commandBufferID,
		oGrad,
		aGrad,
		tgtBufferID,
		smxBufferID,
		C.uint(chunkSize),
	)
}

func CustomKernelCrossEntropyPos(
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

func CustomKernelCrossEntropyPosBwd(
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
