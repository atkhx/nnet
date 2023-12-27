package nllpos

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* nllKernelCreate(void *device, const char *kernelSource) {
    return [[NegLogLikelihoodKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void nllPosForward(
    void *kernel,
    void *commandBuffer,
    void *softmax,
    void *output,
    void *targets,
    uint chunkSize
) {
    [(__bridge NegLogLikelihoodKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        softmax:(id<MTLBuffer>)softmax
        output:(id<MTLBuffer>)output
        targets:(id<MTLBuffer>)targets
        chunkSize:(uint)chunkSize
	];
}

void nllPosBackward(
    void *kernel,
    void *commandBuffer,
    void *outputData,
    void *outputGrad,
    void *targets,
    void *softmax,
    void *nllGrad,
    uint chunkSize
) {
    [(__bridge NegLogLikelihoodKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        targets:(id<MTLBuffer>)targets
        softmax:(id<MTLBuffer>)softmax
        nllGrad:(id<MTLBuffer>)nllGrad
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
		kernelID: C.nllKernelCreate(deviceID, cKernelString),
	}
}

type Kernel struct {
	deviceID unsafe.Pointer
	kernelID unsafe.Pointer
}

func (k *Kernel) Forward(
	commandBufferID,
	softmax,
	output,
	targets unsafe.Pointer,
	chunkSize int,
) {
	C.nllPosForward(
		k.kernelID,
		commandBufferID,
		softmax,
		output,
		targets,
		C.uint(chunkSize),
	)
}

func (k *Kernel) Backward(
	commandBufferID,
	outputData unsafe.Pointer,
	outputGrad unsafe.Pointer,
	targets unsafe.Pointer,
	softmax unsafe.Pointer,
	nllGrad unsafe.Pointer,
	chunkSize int,
) {
	C.nllPosBackward(
		k.kernelID,
		commandBufferID,
		outputData,
		outputGrad,
		targets,
		softmax,
		nllGrad,
		C.uint(chunkSize),
	)
}
