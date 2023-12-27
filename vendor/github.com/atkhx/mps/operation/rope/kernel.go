package rope

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* ropeKernelCreate(void *device, const char *kernelSource) {
    return [[RopeKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void ropeForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    uint headIndex,
    uint headSize,
    uint contextLength

) {

    [(__bridge RopeKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
		headIndex:(uint)headIndex
        headSize:(uint)headSize
        contextLength:(uint)contextLength
	];
}

void ropeBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *outputGrad,
    uint headIndex,
    uint headSize,
    uint contextLength
) {
    [(__bridge RopeKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
		headIndex:(uint)headIndex
        headSize:(uint)headSize
        contextLength:(uint)contextLength
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
		kernelID: C.ropeKernelCreate(deviceID, cKernelString),
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
	headIndex int,
	headSize int,
	contextLength int,
) {
	C.ropeForward(
		k.kernelID,
		commandBufferID,
		inputData,
		outputData,
		C.uint(headIndex),
		C.uint(headSize),
		C.uint(contextLength),
	)
}

func (k *Kernel) Backward(
	commandBufferID unsafe.Pointer,
	inputGrad unsafe.Pointer,
	outputGrad unsafe.Pointer,
	headIndex int,
	headSize int,
	contextLength int,
) {
	C.ropeBackward(
		k.kernelID,
		commandBufferID,
		inputGrad,
		outputGrad,
		C.uint(headIndex),
		C.uint(headSize),
		C.uint(contextLength),
	)
}
