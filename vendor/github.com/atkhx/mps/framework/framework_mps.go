package framework

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders
#include "framework_mps.h"
*/
import "C"
import (
	"unsafe"
)

// MPSMatrixDescriptor

func MPSMatrixDescriptorCreate(cols, rows, batchSize, batchStride int) unsafe.Pointer {
	return C.mpsMatrixDescriptorCreate(C.int(cols), C.int(rows), C.int(batchSize), C.int(batchStride))
}

func MPSMatrixDescriptorRelease(descriptorID unsafe.Pointer) {
	C.mpsMatrixDescriptorRelease(descriptorID)
}

// MPSMatrix

func MPSMatrixCreate(bufferID, descriptorID unsafe.Pointer, offset int) unsafe.Pointer {
	return C.mpsMatrixCreate(bufferID, descriptorID, C.int(offset))
}

func MPSMatrixRelease(matrixID unsafe.Pointer) {
	C.mpsMatrixRelease(matrixID)
}

// MPSMatrixMultiplication

func MPSMatrixMultiplicationCreate(
	deviceID unsafe.Pointer,

	resultRows,
	resultColumns,
	interiorColumns int,

	alpha,
	beta float32,

	transposeLeft,
	transposeRight bool,
) unsafe.Pointer {
	return unsafe.Pointer(C.mpsMatrixMultiplicationCreate(
		deviceID,

		C.int(resultRows),
		C.int(resultColumns),
		C.int(interiorColumns),

		C.float(alpha),
		C.float(beta),

		C.bool(transposeLeft),
		C.bool(transposeRight),
	))
}

func MPSMatrixMultiplicationEncode(commandBufferID, kernelID, aMatrixID, bMatrixID, cMatrixID unsafe.Pointer) {
	C.mpsMatrixMultiplicationEncode(commandBufferID, kernelID, aMatrixID, bMatrixID, cMatrixID)
}

// MPSMatrixRandomDistributionDescriptor

func MPSMatrixRandomDistributionDescriptorCreate(min, max float32) unsafe.Pointer {
	// todo add release func
	return C.mpsMatrixRandomDistributionDescriptorCreate(C.float(min), C.float(max))
}

// MPSMatrixRandomMTGP32

func MPSMatrixRandomMTGP32Create(deviceID, distributionID unsafe.Pointer, seed uint64) unsafe.Pointer {
	// todo add release func
	return C.mpsMatrixRandomMTGP32Create(deviceID, distributionID, C.ulong(seed))
}

func MPSMatrixRandomMTGP32Encode(kernelID, commandBufferID, aMatrixID unsafe.Pointer) {
	C.mpsMatrixRandomMTGP32Encode(kernelID, commandBufferID, aMatrixID)
}

// MPSMatrixSoftMax

func MPSMatrixSoftMaxCreate(deviceID unsafe.Pointer) unsafe.Pointer {
	return unsafe.Pointer(C.mpsMatrixSoftMaxCreate(deviceID))
}

func MPSMatrixSoftMaxEncode(commandBufferID, kernelID, inputMatrix, resultMatrix unsafe.Pointer) {
	C.mpsMatrixSoftMaxEncode(commandBufferID, kernelID, inputMatrix, resultMatrix)
}

// MPSMatrixSoftMaxGradient

func MPSMatrixSoftMaxGradientCreate(deviceID unsafe.Pointer) unsafe.Pointer {
	return unsafe.Pointer(C.mpsMatrixSoftMaxGradientCreate(deviceID))
}

func MPSMatrixSoftMaxGradientEncode(commandBufferID, kernelID, gradientMatrix, forwardOutputMatrix, resultMatrix unsafe.Pointer) {
	C.mpsMatrixSoftMaxGradientEncode(commandBufferID, kernelID, gradientMatrix, forwardOutputMatrix, resultMatrix)
}
