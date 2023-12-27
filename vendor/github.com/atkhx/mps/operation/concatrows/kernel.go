package concatrows

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* concatRowsKernelCreate(void *device, const char *kernelSource) {
    return [[ConcatRowsKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void concatRowsForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    uint inputWidth,
    uint outputWidth,
    uint outputOffset
) {

    [(__bridge ConcatRowsKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        inputWidth:(uint)inputWidth
        outputWidth:(uint)outputWidth
        outputOffset:(uint)outputOffset
	];
}

void concatRowsBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *outputGrad,
    uint inputWidth,
    uint outputWidth,
    uint outputOffset
) {
    [(__bridge ConcatRowsKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        inputWidth:(uint)inputWidth
        outputWidth:(uint)outputWidth
        outputOffset:(uint)outputOffset
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
		kernelID: C.concatRowsKernelCreate(deviceID, cKernelString),
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
	inputWidth int,
	outputWidth int,
	outputOffset int,
) {
	C.concatRowsForward(
		k.kernelID,
		commandBufferID,
		inputData,
		outputData,
		C.uint(inputWidth),
		C.uint(outputWidth),
		C.uint(outputOffset),
	)
}

func (k *Kernel) Backward(
	commandBufferID unsafe.Pointer,
	inputGrad unsafe.Pointer,
	outputGrad unsafe.Pointer,
	inputWidth int,
	outputWidth int,
	outputOffset int,
) {
	C.concatRowsBackward(
		k.kernelID,
		commandBufferID,
		inputGrad,
		outputGrad,
		C.uint(inputWidth),
		C.uint(outputWidth),
		C.uint(outputOffset),
	)
}
