package dropout

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* dropoutKernelCreate(void *device, const char *kernelSource) {
    return [[DropoutKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void dropoutForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    void *randomData,
	float probability
) {
    [(__bridge DropoutKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
		inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        randomData:(id<MTLBuffer>)randomData
		probability:(float)probability
	];
}

void dropoutBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *outputGrad,
    void *randomData,
	float probability
) {
    [(__bridge DropoutKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
		inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        randomData:(id<MTLBuffer>)randomData
		probability:(float)probability
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
		kernelID: C.dropoutKernelCreate(deviceID, cKernelString),
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
	randomData unsafe.Pointer,
	probability float32,
) {
	C.dropoutForward(
		k.kernelID,
		commandBufferID,
		inputData,
		outputData,
		randomData,
		C.float(probability),
	)
}

func (k *Kernel) Backward(
	commandBufferID unsafe.Pointer,
	inputGrad unsafe.Pointer,
	outputGrad unsafe.Pointer,
	randomData unsafe.Pointer,
	probability float32,
) {
	C.dropoutBackward(
		k.kernelID,
		commandBufferID,
		inputGrad,
		outputGrad,
		randomData,
		C.float(probability),
	)
}
