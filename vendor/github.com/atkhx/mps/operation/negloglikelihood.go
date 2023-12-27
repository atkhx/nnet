package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/nllpos"
)

func NewOpNegLogLikelihood(device *mps.MTLDevice, inputData, inputGrad, outputData, outputGrad, targets *mps.MTLBuffer, chunkSize int) *OpNegLogLikelihood {
	return &OpNegLogLikelihood{
		kernel:     nllpos.New(device.DeviceID),
		inputData:  inputData.BufferID,
		inputGrad:  inputGrad.BufferID,
		outputData: outputData.BufferID,
		outputGrad: outputGrad.BufferID,
		targets:    targets.BufferID,
		chunkSize:  chunkSize,
	}
}

type OpNegLogLikelihood struct {
	kernel     *nllpos.Kernel
	inputData  unsafe.Pointer
	inputGrad  unsafe.Pointer
	outputData unsafe.Pointer
	outputGrad unsafe.Pointer
	targets    unsafe.Pointer

	chunkSize int
}

func (op *OpNegLogLikelihood) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Forward(
			b.ID,
			op.inputData,
			op.outputData,
			op.targets,
			op.chunkSize,
		)
	})
}

func (op *OpNegLogLikelihood) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(
			b.ID,
			op.outputData,
			op.outputGrad,
			op.targets,
			op.inputData,
			op.inputGrad,
			op.chunkSize,
		)
	})
}
