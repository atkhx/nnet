package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/mulequal"
)

func NewOpMulEqual(
	device *mps.MTLDevice,
	inputData *mps.MTLBuffer,
	inputGrad *mps.MTLBuffer,
	weightsData *mps.MTLBuffer,
	weightsGrad *mps.MTLBuffer,
	outputData *mps.MTLBuffer,
	outputGrad *mps.MTLBuffer,
) *OpMulEqual {
	return &OpMulEqual{
		kernel: mulequal.New(device.DeviceID),

		inputData:   inputData.BufferID,
		inputGrad:   inputGrad.BufferID,
		weightsData: weightsData.BufferID,
		weightsGrad: weightsGrad.BufferID,
		outputData:  outputData.BufferID,
		outputGrad:  outputGrad.BufferID,
	}
}

type OpMulEqual struct {
	kernel *mulequal.Kernel

	inputData unsafe.Pointer
	inputGrad unsafe.Pointer

	weightsData unsafe.Pointer
	weightsGrad unsafe.Pointer

	outputData unsafe.Pointer
	outputGrad unsafe.Pointer
}

func (op *OpMulEqual) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Forward(
			b.ID,
			op.inputData,
			op.weightsData,
			op.outputData,
		)
	})
}

func (op *OpMulEqual) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(
			b.ID,
			op.inputData,
			op.inputGrad,
			op.weightsData,
			op.weightsGrad,
			op.outputData,
			op.outputGrad,
		)
	})
}
