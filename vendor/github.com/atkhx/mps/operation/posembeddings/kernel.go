package posembeddings

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* posEmbeddingsKernelCreate(void *device, const char *kernelSource) {
    return [[posEmbeddingsKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void posEmbeddingsForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    void *posEmbedding,
    void *tokenEmbedding,
    uint featuresCount,
    uint contextLength
) {
    [(__bridge posEmbeddingsKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        posEmbedding:(id<MTLBuffer>)posEmbedding
        tokenEmbedding:(id<MTLBuffer>)tokenEmbedding
        featuresCount:(uint)featuresCount
        contextLength:(uint)contextLength
	];
}

void posEmbeddingsBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputGrad,
    void *tokenEmbeddingGrad,
	uint featuresCount
) {
    [(__bridge posEmbeddingsKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputGrad:(id<MTLBuffer>)outputGrad
        tokenEmbeddingGrad:(id<MTLBuffer>)tokenEmbeddingGrad
        featuresCount:(uint)featuresCount
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
		kernelID: C.posEmbeddingsKernelCreate(deviceID, cKernelString),
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
	posEmbedding unsafe.Pointer,
	tokenEmbedding unsafe.Pointer,
	featuresCount int,
	contextLength int,
) {
	C.posEmbeddingsForward(
		k.kernelID,
		commandBufferID,
		inputData,
		outputData,
		posEmbedding,
		tokenEmbedding,
		C.uint(featuresCount),
		C.uint(contextLength),
	)
}

func (k *Kernel) Backward(
	commandBufferID unsafe.Pointer,
	inputData unsafe.Pointer,
	outputGrad unsafe.Pointer,
	tokenEmbeddingGrad unsafe.Pointer,
	featuresCount int,
) {
	C.posEmbeddingsBackward(
		k.kernelID,
		commandBufferID,
		inputData,
		outputGrad,
		tokenEmbeddingGrad,
		C.uint(featuresCount),
	)
}
