package transpose

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* transposeKernelCreate(void *device, const char *kernelSource) {
    return [[TransposeKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void transposeForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
	uint width,
	uint height
) {
    [(__bridge TransposeKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
		width:(uint)width
        height:(uint)height
	];
}

void transposeBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *outputGrad,
	uint width,
	uint height

) {
    [(__bridge TransposeKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
		width:(uint)width
        height:(uint)height
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
		kernelID: C.transposeKernelCreate(deviceID, cKernelString),
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
	width int,
	height int,
) {
	C.transposeForward(
		k.kernelID,
		commandBufferID,
		inputData,
		outputData,
		C.uint(width),
		C.uint(height),
	)
}

func (k *Kernel) Backward(
	commandBufferID unsafe.Pointer,
	inputGrad unsafe.Pointer,
	outputGrad unsafe.Pointer,
	width int,
	height int,
) {
	C.transposeBackward(
		k.kernelID,
		commandBufferID,
		inputGrad,
		outputGrad,
		C.uint(width),
		C.uint(height),
	)
}
