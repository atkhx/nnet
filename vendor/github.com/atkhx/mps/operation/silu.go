package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/silu"
)

func NewOpSiLu(device *mps.MTLDevice, inputData, inputGrad, outputData, outputGrad *mps.MTLBuffer) *OpSiLu {
	return &OpSiLu{
		kernel:     silu.New(device.DeviceID),
		inputData:  inputData.BufferID,
		inputGrad:  inputGrad.BufferID,
		outputData: outputData.BufferID,
		outputGrad: outputGrad.BufferID,
	}
}

type OpSiLu struct {
	kernel *silu.Kernel

	inputData unsafe.Pointer
	inputGrad unsafe.Pointer

	outputData unsafe.Pointer
	outputGrad unsafe.Pointer
}

func (op *OpSiLu) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Forward(b.ID, op.inputData, op.outputData)
	})
}

func (op *OpSiLu) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(b.ID, op.inputData, op.inputGrad, op.outputData, op.outputGrad)
	})
}
