package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/relu"
)

func NewOpReLu(device *mps.MTLDevice, inputData, inputGrad, outputData, outputGrad *mps.MTLBuffer) *OpReLu {
	return &OpReLu{
		kernel:     relu.New(device.DeviceID),
		inputData:  inputData.BufferID,
		inputGrad:  inputGrad.BufferID,
		outputData: outputData.BufferID,
		outputGrad: outputGrad.BufferID,
	}
}

type OpReLu struct {
	kernel *relu.Kernel

	inputData unsafe.Pointer
	inputGrad unsafe.Pointer

	outputData unsafe.Pointer
	outputGrad unsafe.Pointer
}

func (op *OpReLu) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Forward(b.ID, op.inputData, op.outputData)
	})
}

func (op *OpReLu) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(b.ID, op.inputData, op.inputGrad, op.outputGrad)
	})
}
