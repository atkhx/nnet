package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/rope"
)

func NewOpRope(
	device *mps.MTLDevice,
	inputData *mps.MTLBuffer,
	inputGrad *mps.MTLBuffer,
	outputData *mps.MTLBuffer,
	outputGrad *mps.MTLBuffer,
	headIndex int,
	headSize int,
	contextLength int,
) *OpRoPE {
	return &OpRoPE{
		kernel:        rope.New(device.DeviceID),
		inputData:     inputData.BufferID,
		inputGrad:     inputGrad.BufferID,
		outputData:    outputData.BufferID,
		outputGrad:    outputGrad.BufferID,
		headIndex:     headIndex,
		headSize:      headSize,
		contextLength: contextLength,
	}
}

type OpRoPE struct {
	kernel *rope.Kernel

	inputData unsafe.Pointer
	inputGrad unsafe.Pointer

	outputData unsafe.Pointer
	outputGrad unsafe.Pointer

	headIndex     int
	headSize      int
	contextLength int
}

func (op *OpRoPE) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Forward(
			b.ID,
			op.inputData,
			op.outputData,
			op.headIndex,
			op.headSize,
			op.contextLength,
		)
	})
}

func (op *OpRoPE) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(
			b.ID,
			op.inputGrad,
			op.outputGrad,
			op.headIndex,
			op.headSize,
			op.contextLength,
		)
	})
}
