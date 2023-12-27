package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/posembeddings"
)

func NewOpPosEmbeddings(
	device *mps.MTLDevice,
	tokenEmbeddingData,
	tokenEmbeddingGrad,
	positionEmbeddingData,
	inputData,
	outputData,
	outputGrad *mps.MTLBuffer,
	featuresCount, contextLength int,
) *OpPosEmbeddings {
	return &OpPosEmbeddings{
		kernel:                posembeddings.New(device.DeviceID),
		tokenEmbeddingData:    tokenEmbeddingData.BufferID,
		tokenEmbeddingGrad:    tokenEmbeddingGrad.BufferID,
		positionEmbeddingData: positionEmbeddingData.BufferID,
		inputData:             inputData.BufferID,
		outputData:            outputData.BufferID,
		outputGrad:            outputGrad.BufferID,
		featuresCount:         featuresCount,
		contextLength:         contextLength,
	}
}

type OpPosEmbeddings struct {
	kernel *posembeddings.Kernel

	tokenEmbeddingData unsafe.Pointer
	tokenEmbeddingGrad unsafe.Pointer

	positionEmbeddingData unsafe.Pointer

	inputData  unsafe.Pointer
	outputData unsafe.Pointer
	outputGrad unsafe.Pointer

	featuresCount int
	contextLength int
}

func (op *OpPosEmbeddings) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Forward(
			b.ID,
			op.inputData,
			op.outputData,
			op.positionEmbeddingData,
			op.tokenEmbeddingData,
			op.featuresCount,
			op.contextLength,
		)
	})
}

func (op *OpPosEmbeddings) Backward(b *mps.MTLCommandBuffer) {
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
