package mulrows

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* mulRowsKernelCreate(void *device, const char *kernelSource) {
    return [[MulRowsKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void mulRowsForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *weightsData,
    void *outputData,
	uint chunkSize
) {

    [(__bridge MulRowsKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
        chunkSize:(uint)chunkSize
	];
}

void mulRowsBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
	void *weightsData,
	void *weightsGrad,
    void *outputData,
    void *outputGrad,
	uint chunkSize
) {
    [(__bridge MulRowsKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsData:(id<MTLBuffer>)weightsData
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputData:(id<MTLBuffer>)outputData
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
		kernelID: C.mulRowsKernelCreate(deviceID, cKernelString),
	}
}

type Kernel struct {
	deviceID unsafe.Pointer
	kernelID unsafe.Pointer
}

func (k *Kernel) Forward(
	commandBufferID unsafe.Pointer,
	inputData unsafe.Pointer,
	weightsData unsafe.Pointer,
	outputData unsafe.Pointer,
	chunkSize int,
) {
	C.mulRowsForward(
		k.kernelID,
		commandBufferID,
		inputData,
		weightsData,
		outputData,
		C.uint(chunkSize),
	)
}

func (k *Kernel) Backward(
	commandBufferID unsafe.Pointer,
	inputData unsafe.Pointer,
	inputGrad unsafe.Pointer,
	weightsData unsafe.Pointer,
	weightsGrad unsafe.Pointer,
	outputData unsafe.Pointer,
	outputGrad unsafe.Pointer,
	chunkSize int,
) {
	C.mulRowsBackward(
		k.kernelID,
		commandBufferID,
		inputData,
		inputGrad,
		weightsData,
		weightsGrad,
		outputData,
		outputGrad,
		C.uint(chunkSize),
	)
}
