package addequal

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* addEqualKernelCreate(void *device, const char *kernelSource) {
    return [[AddEqualKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void addEqualForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *weightsData,
    void *outputData
) {
    [(__bridge AddEqualKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
	];
}

void addEqualBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
	void *weightsGrad,
    void *outputGrad
) {
    [(__bridge AddEqualKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputGrad:(id<MTLBuffer>)outputGrad
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
		kernelID: C.addEqualKernelCreate(deviceID, cKernelString),
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
) {
	C.addEqualForward(
		k.kernelID,
		commandBufferID,
		inputData,
		weightsData,
		outputData,
	)
}

func (k *Kernel) Backward(
	commandBufferID unsafe.Pointer,
	inputGrad unsafe.Pointer,
	weightsGrad unsafe.Pointer,
	outputGrad unsafe.Pointer,
) {
	C.addEqualBackward(
		k.kernelID,
		commandBufferID,
		inputGrad,
		weightsGrad,
		outputGrad,
	)
}
