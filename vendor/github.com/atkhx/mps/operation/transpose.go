package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/transpose"
)

func NewOpTranspose(device *mps.MTLDevice, inputData, inputGrad, outputData, outputGrad *mps.MTLBuffer, width, height int) *OpTranspose {
	return &OpTranspose{
		kernel:     transpose.New(device.DeviceID),
		inputData:  inputData.BufferID,
		inputGrad:  inputGrad.BufferID,
		outputData: outputData.BufferID,
		outputGrad: outputGrad.BufferID,
		width:      width,
		height:     height,
	}
}

type OpTranspose struct {
	kernel *transpose.Kernel

	inputData unsafe.Pointer
	inputGrad unsafe.Pointer

	outputData unsafe.Pointer
	outputGrad unsafe.Pointer

	width  int
	height int
}

func (op *OpTranspose) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Forward(
			b.ID,
			op.inputData,
			op.outputData,
			op.width,
			op.height,
		)
	})
}

func (op *OpTranspose) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(
			b.ID,
			op.inputGrad,
			op.outputGrad,
			op.width,
			op.height,
		)
	})
}
