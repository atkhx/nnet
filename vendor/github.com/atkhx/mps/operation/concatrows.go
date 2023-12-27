package operation

import (
	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/concatrows"
)

func NewOpConcatByRows(device *mps.MTLDevice, inputData, inputGrad []*mps.MTLBuffer, outputData, outputGrad *mps.MTLBuffer, inputWidth int) *OpConcatByRows {
	return &OpConcatByRows{
		kernel:     concatrows.New(device.DeviceID),
		inputData:  inputData,
		inputGrad:  inputGrad,
		outputData: outputData,
		outputGrad: outputGrad,
		inputWidth: inputWidth,
	}
}

type OpConcatByRows struct {
	kernel *concatrows.Kernel

	inputData []*mps.MTLBuffer
	inputGrad []*mps.MTLBuffer

	outputData *mps.MTLBuffer
	outputGrad *mps.MTLBuffer

	inputWidth int
}

func (op *OpConcatByRows) Forward(b *mps.MTLCommandBuffer) {
	inputWidth := op.inputWidth
	outputWidth := inputWidth * len(op.inputData)

	b.Exclusive(func() {
		for i, inputData := range op.inputData {
			op.kernel.Forward(
				b.ID,
				inputData.BufferID,
				op.outputData.BufferID,
				inputWidth,
				outputWidth,
				i*inputWidth, // outputData offset
			)
		}
	})
}

func (op *OpConcatByRows) Backward(b *mps.MTLCommandBuffer) {
	inputWidth := op.inputWidth
	outputWidth := inputWidth * len(op.inputData)

	b.Exclusive(func() {
		for i, inputGrad := range op.inputGrad {
			op.kernel.Backward(
				b.ID,
				inputGrad.BufferID,
				op.outputGrad.BufferID,
				inputWidth,
				outputWidth,
				i*inputWidth, // outputData offset
			)
		}
	})
}
