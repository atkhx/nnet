package mean

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* meanKernelCreate(void *device, const char *kernelSource) {
    return [[MeanKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void meanForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
	uint chunkSize
) {
    [(__bridge MeanKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        chunkSize:(uint)chunkSize
	];
}

void meanBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *outputGrad,
	uint chunkSize
) {
    [(__bridge MeanKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        chunkSize:(uint)chunkSize
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
		kernelID: C.meanKernelCreate(deviceID, cKernelString),
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
	chunkSize int,
) {
	C.meanForward(
		k.kernelID,
		commandBufferID,
		inputData,
		outputData,
		C.uint(chunkSize),
	)
}

func (k *Kernel) Backward(
	commandBufferID unsafe.Pointer,
	inputGrad unsafe.Pointer,
	outputGrad unsafe.Pointer,
	chunkSize int,
) {
	C.meanBackward(
		k.kernelID,
		commandBufferID,
		inputGrad,
		outputGrad,
		C.uint(chunkSize),
	)
}
