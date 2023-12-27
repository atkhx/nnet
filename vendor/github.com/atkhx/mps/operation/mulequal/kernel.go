package mulequal

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* mulEqualKernelCreate(void *device, const char *kernelSource) {
    return [[MulEqualKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void mulEqualForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *weightsData,
    void *outputData
) {

    [(__bridge MulEqualKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
	];
}

void mulEqualBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
	void *weightsData,
	void *weightsGrad,
    void *outputData,
    void *outputGrad
) {
    [(__bridge MulEqualKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsData:(id<MTLBuffer>)weightsData
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputData:(id<MTLBuffer>)outputData
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
		kernelID: C.mulEqualKernelCreate(deviceID, cKernelString),
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
	C.mulEqualForward(
		k.kernelID,
		commandBufferID,
		inputData,
		weightsData,
		outputData,
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
) {
	C.mulEqualBackward(
		k.kernelID,
		commandBufferID,
		inputData,
		inputGrad,
		weightsData,
		weightsGrad,
		outputData,
		outputGrad,
	)
}
