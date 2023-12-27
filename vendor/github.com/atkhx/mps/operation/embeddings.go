package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/embeddings"
)

func NewOpEmbeddings(
	device *mps.MTLDevice,
	tokenEmbeddingData,
	tokenEmbeddingGrad,
	inputData,
	outputData,
	outputGrad *mps.MTLBuffer,
	featuresCount, contextLength int,
) *OpEmbeddings {
	return &OpEmbeddings{
		kernel:             embeddings.New(device.DeviceID),
		tokenEmbeddingData: tokenEmbeddingData.BufferID,
		tokenEmbeddingGrad: tokenEmbeddingGrad.BufferID,
		inputData:          inputData.BufferID,
		outputData:         outputData.BufferID,
		outputGrad:         outputGrad.BufferID,
		featuresCount:      featuresCount,
		contextLength:      contextLength,
	}
}

type OpEmbeddings struct {
	kernel *embeddings.Kernel

	tokenEmbeddingData unsafe.Pointer
	tokenEmbeddingGrad unsafe.Pointer

	inputData  unsafe.Pointer
	outputData unsafe.Pointer
	outputGrad unsafe.Pointer

	featuresCount int
	contextLength int
}

func (op *OpEmbeddings) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Forward(
			b.ID,
			op.inputData,
			op.outputData,
			op.tokenEmbeddingData,
			op.featuresCount,
			op.contextLength,
		)
	})
}

func (op *OpEmbeddings) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(
			b.ID,
			op.inputData,
			op.outputGrad,
			op.tokenEmbeddingGrad,
			op.featuresCount,
		)
	})
}
