package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/framework"
)

func NewOpSoftmax(device *mps.MTLDevice, inputData, inputGrad, outputData, outputGrad *mps.MTLBuffer, chunkSize int) *OpSoftmax {
	inputHeight := inputData.Length / chunkSize

	return &OpSoftmax{
		softmaxKernelID:         framework.MPSMatrixSoftMaxCreate(device.DeviceID),
		softmaxGradientKernelID: framework.MPSMatrixSoftMaxGradientCreate(device.DeviceID),

		inputDataMatrixID:  inputData.CreateMatrixBatch(chunkSize, inputHeight, 1, chunkSize*inputHeight, 0).MatrixID,
		inputGradMatrixID:  inputGrad.CreateMatrixBatch(chunkSize, inputHeight, 1, chunkSize*inputHeight, 0).MatrixID,
		outputDataMatrixID: outputData.CreateMatrixBatch(chunkSize, inputHeight, 1, chunkSize*inputHeight, 0).MatrixID,
		outputGradMatrixID: outputGrad.CreateMatrixBatch(chunkSize, inputHeight, 1, chunkSize*inputHeight, 0).MatrixID,

		chunkSize: chunkSize,
	}
}

type OpSoftmax struct {
	softmaxKernelID         unsafe.Pointer
	softmaxGradientKernelID unsafe.Pointer

	inputDataMatrixID  unsafe.Pointer
	inputGradMatrixID  unsafe.Pointer
	outputDataMatrixID unsafe.Pointer
	outputGradMatrixID unsafe.Pointer

	chunkSize int
}

func (op *OpSoftmax) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		framework.MPSMatrixSoftMaxEncode(
			b.ID,
			op.softmaxKernelID,
			op.inputDataMatrixID,
			op.outputDataMatrixID,
		)
	})
}

func (op *OpSoftmax) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		framework.MPSMatrixSoftMaxGradientEncode(
			b.ID,
			op.softmaxGradientKernelID,
			op.outputGradMatrixID,
			op.outputDataMatrixID,
			op.inputGradMatrixID,
		)
	})
}
