package embeddings

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* embeddingsKernelCreate(void *device, const char *kernelSource) {
    return [[EmbeddingsKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void embeddingsForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    void *tokenEmbedding,
    uint featuresCount,
    uint contextLength
) {
    [(__bridge EmbeddingsKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        tokenEmbedding:(id<MTLBuffer>)tokenEmbedding
        featuresCount:(uint)featuresCount
        contextLength:(uint)contextLength
	];
}

void embeddingsBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputGrad,
    void *tokenEmbeddingGrad,
	uint featuresCount
) {
    [(__bridge EmbeddingsKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
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
		kernelID: C.embeddingsKernelCreate(deviceID, cKernelString),
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
	tokenEmbedding unsafe.Pointer,
	featuresCount int,
	contextLength int,
) {
	C.embeddingsForward(
		k.kernelID,
		commandBufferID,
		inputData,
		outputData,
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
	C.embeddingsBackward(
		k.kernelID,
		commandBufferID,
		inputData,
		outputGrad,
		tokenEmbeddingGrad,
		C.uint(featuresCount),
	)
}
