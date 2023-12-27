package addrows

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* addRowsKernelCreate(void *device, const char *kernelSource) {
    return [[AddRowsKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void addRowsForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *weightsData,
    void *outputData,
	uint chunkSize
) {
    [(__bridge AddRowsKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
        chunkSize:(uint)chunkSize
	];
}

void addRowsBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
	void *weightsGrad,
    void *outputGrad,
	uint chunkSize
) {
    [(__bridge AddRowsKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsGrad:(id<MTLBuffer>)weightsGrad
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
		kernelID: C.addRowsKernelCreate(deviceID, cKernelString),
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
	C.addRowsForward(
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
	inputGrad unsafe.Pointer,
	weightsGrad unsafe.Pointer,
	outputGrad unsafe.Pointer,
	chunkSize int,
) {
	C.addRowsBackward(
		k.kernelID,
		commandBufferID,
		inputGrad,
		weightsGrad,
		outputGrad,
		C.uint(chunkSize),
	)
}
