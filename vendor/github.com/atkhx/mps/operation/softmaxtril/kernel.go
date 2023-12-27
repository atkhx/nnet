package softmaxtril

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* softmaxtrilKernelCreate(void *device, const char *kernelSource) {
    return [[SoftmaxtrilKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void softmaxtrilForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
	uint colsCount,
	uint rowsCount,
	uint offset
) {

    [(__bridge SoftmaxtrilKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
	];
}

void softmaxtrilBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *outputGrad,
    void *outputData,
	uint colsCount,
	uint rowsCount,
	uint offset

) {
    [(__bridge SoftmaxtrilKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        outputData:(id<MTLBuffer>)outputData
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
	];
}

*/
import "C"
import (
	_ "embed"
	"unsafe"
)

//go:embed kernel.metal
var metalFunctions string

func New(deviceID unsafe.Pointer) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		deviceID: deviceID,
		kernelID: C.softmaxtrilKernelCreate(deviceID, cKernelString),
	}
}

type Kernel struct {
	deviceID unsafe.Pointer
	kernelID unsafe.Pointer
}

func (k *Kernel) Forward(
	commandBufferID unsafe.Pointer,
	inputData unsafe.Pointer,
	outputData unsafe.Pointer,
	colsCount int,
	rowsCount int,
	offset int,
) {
	C.softmaxtrilForward(
		k.kernelID,
		commandBufferID,
		inputData,
		outputData,
		C.uint(colsCount),
		C.uint(rowsCount),
		C.uint(offset),
	)
}

func (k *Kernel) Backward(
	commandBufferID unsafe.Pointer,
	inputGrad unsafe.Pointer,
	outputGrad unsafe.Pointer,
	outputData unsafe.Pointer,
	colsCount int,
	rowsCount int,
	offset int,
) {
	C.softmaxtrilBackward(
		k.kernelID,
		commandBufferID,
		inputGrad,
		outputGrad,
		outputData,
		C.uint(colsCount),
		C.uint(rowsCount),
		C.uint(offset),
	)
}
