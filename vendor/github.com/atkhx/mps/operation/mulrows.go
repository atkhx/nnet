package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/mulrows"
)

func NewOpMulRows(
	device *mps.MTLDevice,
	inputData *mps.MTLBuffer,
	inputGrad *mps.MTLBuffer,
	weightsData *mps.MTLBuffer,
	weightsGrad *mps.MTLBuffer,
	outputData *mps.MTLBuffer,
	outputGrad *mps.MTLBuffer,
	chunkSize int,
) *OpMulRows {
	return &OpMulRows{
		kernel: mulrows.New(device.DeviceID),

		inputData:   inputData.BufferID,
		inputGrad:   inputGrad.BufferID,
		weightsData: weightsData.BufferID,
		weightsGrad: weightsGrad.BufferID,
		outputData:  outputData.BufferID,
		outputGrad:  outputGrad.BufferID,

		chunkSize: chunkSize,
	}
}

type OpMulRows struct {
	kernel *mulrows.Kernel

	inputData unsafe.Pointer
	inputGrad unsafe.Pointer

	weightsData unsafe.Pointer
	weightsGrad unsafe.Pointer

	outputData unsafe.Pointer
	outputGrad unsafe.Pointer

	chunkSize int
}

func (op *OpMulRows) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Forward(
			b.ID,
			op.inputData,
			op.weightsData,
			op.outputData,
			op.chunkSize,
		)
	})
}

func (op *OpMulRows) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(
			b.ID,
			op.inputData,
			op.inputGrad,
			op.weightsData,
			op.weightsGrad,
			op.outputData,
			op.outputGrad,
			op.chunkSize,
		)
	})
}
