package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/addrows"
)

func NewOpAddRows(
	device *mps.MTLDevice,
	inputData *mps.MTLBuffer,
	inputGrad *mps.MTLBuffer,
	weightsData *mps.MTLBuffer,
	weightsGrad *mps.MTLBuffer,
	outputData *mps.MTLBuffer,
	outputGrad *mps.MTLBuffer,
	chunkSize int,
) *OpAddRows {
	return &OpAddRows{
		kernel: addrows.New(device.DeviceID),

		inputData:   inputData.BufferID,
		inputGrad:   inputGrad.BufferID,
		weightsData: weightsData.BufferID,
		weightsGrad: weightsGrad.BufferID,
		outputData:  outputData.BufferID,
		outputGrad:  outputGrad.BufferID,

		chunkSize: chunkSize,
	}
}

type OpAddRows struct {
	kernel *addrows.Kernel

	inputData unsafe.Pointer
	inputGrad unsafe.Pointer

	weightsData unsafe.Pointer
	weightsGrad unsafe.Pointer

	outputData unsafe.Pointer
	outputGrad unsafe.Pointer

	chunkSize int
}

func (op *OpAddRows) Forward(b *mps.MTLCommandBuffer) {
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

func (op *OpAddRows) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(
			b.ID,
			op.inputGrad,
			op.weightsGrad,
			op.outputGrad,
			op.chunkSize,
		)
	})
}
